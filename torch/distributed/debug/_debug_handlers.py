# [MODULE DOC] Concrete DebugHandler implementations for the debug frontend (index, stack traces,
# py-spy, FlightRecorder, profiler, TCPStore, and others) with their Jinja templates, plus
# default_handlers. Requires tabulate and the flight recorder components at import time.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tabulate import tabulate

from torch.distributed.debug._frontend import (
    DebugHandler,
    fetch_all,
    format_fetch_summary,
    format_json,
    NavLink,
    Response,
    Route,
)
from torch.distributed.debug._store import tcpstore_client
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.config_manager import JobConfig
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Database,
    Group,
    Membership,
    NCCLCall,
)


if TYPE_CHECKING:
    from torch.distributed.debug._frontend import FrontendServer, HTTPRequestHandler


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler-specific templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
  <h1>{% block title %}Index{% endblock %}</h1>
{% endblock %}
{% block content %}
Hi
{% endblock %}
    """

PYSPY_DUMP_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}py-spy Stack Traces{% endblock %}</h1>
{% endblock %}
{% block content %}
    <form action="" method="get">
        <input type="checkbox" id="native" name="native" value="1"/>
        <label for="native">Native</label>
        <input type="checkbox" id="subprocesses" name="subprocesses" value="1"/>
        <label for="subprocesses">Subprocesses</label>
        <input type="submit" value="Submit">
    </form>

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

FR_TRACE_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}{{ title }}{% endblock %}</h1>
{% endblock %}
{% block content %}
    {% if fetch_summary %}<pre>{{ fetch_summary }}</pre>{% endif %}
    <h2>Groups</h2>
    {{ groups | safe }}
    <h2>Memberships</h2>
    {{ memberships | safe }}
    <h2>Collectives</h2>
    {{ collectives | safe }}
    <h2>NCCL Calls</h2>
    {{ ncclcalls | safe }}
{% endblock %}
    """

PROFILE_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}torch.profiler{% endblock %}</h1>
{% endblock %}

{% block content %}
    <form action="" method="get">
        <label for="duration">Duration (seconds):</label>
        <input type="number" id="duration" name="duration" value="{{ duration }}" min="1" max="60">
        <input type="submit" value="Submit">
    </form>

    <script>
    function stringToArrayBuffer(str) {
        const encoder = new TextEncoder();
        return encoder.encode(str).buffer;
    }
    async function openPerfetto(data) {
        const ui = window.open('https://ui.perfetto.dev/#!/');
        if (!ui) { alert('Popup blocked. Allow popups for this page and click again.'); return; }

        // Perfetto readiness handshake: PING until we receive PONG
        await new Promise((resolve, reject) => {
        const onMsg = (e) => {
            if (e.source === ui && e.data === 'PONG') {
            window.removeEventListener('message', onMsg);
            clearInterval(pinger);
            resolve();
            }
        };
        window.addEventListener('message', onMsg);
        const pinger = setInterval(() => { try { ui.postMessage('PING', '*'); } catch (_e) {} }, 250);
        setTimeout(() => { clearInterval(pinger); window.removeEventListener('message', onMsg); reject(); }, 20000);
        }).catch(() => { alert('Perfetto UI did not respond. Try again.'); return; });

        ui.postMessage({
        perfetto: {
            buffer: stringToArrayBuffer(JSON.stringify(data)),
            title: "torch profiler",
            fileName: "trace.json",
        }
        }, '*');
    }
    </script>

    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h2>Rank {{ i }}: {{ addr }}</h2>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <script>
            function run{{ i }}() {
                var data = {{ resp.text | safe }};
                openPerfetto(data);
            }
            </script>

            <button onclick="run{{ i }}()">View {{ i }}</button>
        {% endif %}
    {% endfor %}
{% endblock %}
    """

TCPSTORE_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}TCPStore Keys{% endblock %}</h1>
{% endblock %}
{% block content %}
    <pre>
    {% for k, v in zip(keys, values) -%}
{{ k }}: {{ v | truncate(100) }}
    {% endfor %}
    </pre>
{% endblock %}
    """


# ---------------------------------------------------------------------------
# Handler classes
# ---------------------------------------------------------------------------


class IndexHandler(DebugHandler):
    # [CLASS DOC] Serves the home page at the root path with a Home nav link.
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the single route for the root path.
        return [Route("/", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the Home nav link.
        return [NavLink("/", "Home")]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the index template.
        return {"index.html": INDEX_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Renders the static index page.
        return req.frontend.render_template("index.html")


class StacksHandler(DebugHandler):
    # [CLASS DOC] Shows Python stack traces from all ranks at the stacks path, and supports
    # periodic dumps under the file name "stacks".
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the route for the stacks path.
        return [Route("/stacks", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the Python Stack Traces nav link.
        return [NavLink("/stacks", "Python Stack Traces")]

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches the traceback dump from every rank and renders the raw page.
        addrs, resps = fetch_all("dump_traceback", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "raw_resp.html", title="Stacks", addrs=addrs, resps=resps
        )

    def dump(self) -> str | None:
        # [FUNCTION DOC] Fetches tracebacks from all ranks and returns text with an optional partial
        # data summary followed by a section per rank (body, or an error line for failed ranks).
        addrs, resps = fetch_all("dump_traceback", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "stacks".
        return "stacks"


class PySpyHandler(DebugHandler):
    # [CLASS DOC] Shows py-spy stack dumps from all ranks (optional native and subprocess flags
    # come from the query string) and supports periodic dumps under "pyspy_dump".
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the route for the py-spy dump path.
        return [Route("/pyspy_dump", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the py-spy Stacks nav link.
        return [NavLink("/pyspy_dump", "py-spy Stacks")]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the py-spy page template.
        return {"pyspy_dump.html": PYSPY_DUMP_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Forwards the request query to all ranks, adding the nonblocking flag
        # if absent, and renders the py-spy page with the responses.
        query = req.get_raw_query()
        if "nonblocking" not in query:
            query = f"nonblocking=1&{query}" if query else "nonblocking=1"
        addrs, resps = fetch_all("pyspy_dump", query, timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "pyspy_dump.html",
            addrs=addrs,
            resps=resps,
        )

    def dump(self) -> str | None:
        # [FUNCTION DOC] Fetches non-blocking py-spy dumps from all ranks and returns text with an
        # optional partial data summary and a section per rank (body or error line).
        addrs, resps = fetch_all(
            "pyspy_dump", "nonblocking=1", timeout=self.fetch_timeout
        )
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "pyspy_dump".
        return "pyspy_dump"


class FlightRecorderHandler(DebugHandler):
    # [CLASS DOC] Serves FlightRecorder data from all ranks as analyzed HTML tables (groups,
    # memberships, collectives, NCCL calls) or raw JSON, for both the CPU-side trace and the NCCL
    # active-only trace. Periodic dumps are plain text tables under the handler default name.
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns routes for the trace, trace JSON, NCCL trace and NCCL trace JSON paths.
        return [
            Route("/fr_trace", self._handle_fr_trace),
            Route("/fr_trace_json", self._handle_fr_trace_json),
            Route("/fr_trace_nccl", self._handle_fr_trace_nccl),
            Route("/fr_trace_nccl_json", self._handle_fr_trace_nccl_json),
        ]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns four nav links (CPU, its JSON, NCCL, its JSON).
        return [
            NavLink("/fr_trace", "FlightRecorder CPU"),
            NavLink("/fr_trace_json", "(JSON)"),
            NavLink("/fr_trace_nccl", "FlightRecorder NCCL"),
            NavLink("/fr_trace_nccl_json", "(JSON)"),
        ]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the FlightRecorder tables template.
        return {"fr_trace.html": FR_TRACE_TEMPLATE}

    @staticmethod
    def _build_db(addrs: list[str], resps: list[Response]) -> Database:
        # [FUNCTION DOC] HELPER: Builds a flight recorder Database from per-rank responses. Skips
        # non-200 responses, tags each dump with its rank and host, defaults missing entries to an
        # empty list, and runs the analyzer allowing incomplete ranks. Raises RuntimeError if no rank
        # responded. The version is taken from the first successful dump.
        config = JobConfig()
        args = config.parse_args(args=[])
        args.allow_incomplete_ranks = True
        args.verbose = True

        details = {}
        for rank, resp in enumerate(resps):
            if resp.status_code != 200:
                continue
            dump = {
                "rank": rank,
                "host_name": addrs[rank],
                **resp.json(),
            }
            if "entries" not in dump:
                dump["entries"] = []
            details[f"rank{rank}.json"] = dump

        if not details:
            raise RuntimeError("All workers failed to respond")

        version = next(iter(details.values()))["version"]
        # pyrefly: ignore [bad-argument-type]
        return build_db(details, args, version)

    def _render_tables(
        self, server: FrontendServer, addrs: list[str], resps: list[Response]
    ) -> bytes:
        # [FUNCTION DOC] HELPER: Builds the database and renders the HTML page with the fetch
        # summary and four tables.
        db = self._build_db(addrs, resps)
        return server.render_template(
            "fr_trace.html",
            title="FlightRecorder",
            fetch_summary=format_fetch_summary(addrs, resps),
            groups=tabulate(db.groups, headers=Group._fields, tablefmt="html"),
            memberships=tabulate(
                db.memberships, headers=Membership._fields, tablefmt="html"
            ),
            collectives=tabulate(
                db.collectives, headers=Collective._fields, tablefmt="html"
            ),
            ncclcalls=tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="html"),
        )

    def _handle_fr_trace(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches the FlightRecorder JSON from all ranks and renders tables.
        addrs, resps = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches the FlightRecorder JSON from all ranks and renders it pretty-printed.
        addrs, resps = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "json_resp.html",
            title="FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _handle_fr_trace_nccl(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches active-only NCCL trace JSON from all ranks and renders tables.
        addrs, resps = fetch_all(
            "dump_nccl_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_fr_trace_nccl_json(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches active-only NCCL trace JSON from all ranks and renders it pretty-printed.
        addrs, resps = fetch_all(
            "dump_nccl_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return req.frontend.render_template(
            "json_resp.html",
            title="FlightRecorder NCCL",
            addrs=addrs,
            resps=resps,
        )

    def dump(self) -> str | None:
        # [FUNCTION DOC] Returns plain-text tables (groups, memberships, collectives, NCCL calls) built
        # from the FlightRecorder trace of all ranks, preceded by a partial data summary if any rank
        # failed. Then tries the active-only NCCL trace the same way; any error there is replaced by a
        # failure line. Errors in the first (CPU) trace propagate (for example when all ranks failed).
        parts = []

        addrs, resps = fetch_all("fr_trace_json", timeout=self.fetch_timeout)
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        db = self._build_db(addrs, resps)
        parts.extend(
            [
                "=== FR Trace ===",
                "--- Groups ---",
                tabulate(db.groups, headers=Group._fields, tablefmt="plain"),
                "--- Memberships ---",
                tabulate(db.memberships, headers=Membership._fields, tablefmt="plain"),
                "--- Collectives ---",
                tabulate(db.collectives, headers=Collective._fields, tablefmt="plain"),
                "--- NCCL Calls ---",
                tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="plain"),
            ]
        )

        try:
            nccl_addrs, nccl_resps = fetch_all(
                "dump_nccl_trace_json", "onlyactive=true", timeout=self.fetch_timeout
            )
            nccl_db = self._build_db(nccl_addrs, nccl_resps)
            parts.extend(
                [
                    "",
                    "=== FR Trace NCCL ===",
                    "--- Groups ---",
                    tabulate(nccl_db.groups, headers=Group._fields, tablefmt="plain"),
                    "--- Memberships ---",
                    tabulate(
                        nccl_db.memberships,
                        headers=Membership._fields,
                        tablefmt="plain",
                    ),
                    "--- Collectives ---",
                    tabulate(
                        nccl_db.collectives,
                        headers=Collective._fields,
                        tablefmt="plain",
                    ),
                    "--- NCCL Calls ---",
                    tabulate(
                        nccl_db.ncclcalls,
                        headers=NCCLCall._fields,
                        tablefmt="plain",
                    ),
                ]
            )
        except Exception:
            parts.append("\n=== FR Trace NCCL ===\nFailed to fetch NCCL trace")

        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "fr_trace".
        return "fr_trace"


class ProfilerHandler(DebugHandler):
    # [CLASS DOC] Triggers torch profiling on all ranks for a requested duration and renders a page
    # with a button per rank to open the trace in the Perfetto UI. No periodic dump.
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the route for the profile path.
        return [Route("/profile", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the torch profiler nav link.
        return [NavLink("/profile", "torch profiler")]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the profiler page template.
        return {"profile.html": PROFILE_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Reads the duration query parameter (float, default 1 second), asks all
        # ranks to profile for that long and renders the page with the responses. The page form
        # is not limited server-side; a non-numeric value raises an error (reported as 500).
        duration = req.get_query_arg("duration", default=1.0, type=float)
        addrs, resps = fetch_all(
            "torch_profile", f"duration={duration}", timeout=self.fetch_timeout
        )
        return req.frontend.render_template("profile.html", addrs=addrs, resps=resps)


class WaitCountersHandler(DebugHandler):
    # [CLASS DOC] Shows wait counter values (JSON) from all ranks; supports periodic dumps under
    # "wait_counters".
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the route for the wait counters path.
        return [Route("/wait_counters", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the Wait Counters nav link.
        return [NavLink("/wait_counters", "Wait Counters")]

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches wait counter values from all ranks and renders them as JSON.
        addrs, resps = fetch_all("wait_counter_values", timeout=self.fetch_timeout)
        return req.frontend.render_template(
            "json_resp.html", title="Wait Counters", addrs=addrs, resps=resps
        )

    def dump(self) -> str | None:
        # [FUNCTION DOC] Fetches wait counters from all ranks and returns text with an optional partial
        # data summary and a section per rank (pretty JSON or an error line).
        addrs, resps = fetch_all("wait_counter_values", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            if resp.status_code == 200:
                parts.append(format_json(resp.text))
            else:
                parts.append(f"Error: {resp.status_code}")
        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "wait_counters".
        return "wait_counters"


class TCPStoreHandler(DebugHandler):
    # [CLASS DOC] Lists all keys and values of the job TCPStore (unprefixed), sorted by key; supports
    # periodic dumps under "tcpstore".
    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the route for the tcpstore path.
        return [Route("/tcpstore", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the TCPStore nav link.
        return [NavLink("/tcpstore", "TCPStore")]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the TCPStore page template.
        return {"tcpstore.html": TCPSTORE_TEMPLATE}

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Connects to the store, reads all sorted keys and the textual form of
        # their values, and renders the page.
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        return req.frontend.render_template("tcpstore.html", keys=keys, values=values)

    def dump(self) -> str | None:
        # [FUNCTION DOC] Returns one line per store key (sorted) with the textual form of its value.
        store = tcpstore_client(prefix="")
        keys = store.list_keys()
        keys.sort()
        values = [repr(v) for v in store.multi_get(keys)]
        parts = [f"{k}: {v}" for k, v in zip(keys, values)]
        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "tcpstore".
        return "tcpstore"


class TorchCommsFlightRecorderHandler(DebugHandler):
    # [CLASS DOC] Serves TorchComms FlightRecorder traces (active only) from all ranks as analyzed
    # tables or raw JSON, reusing the FlightRecorderHandler database builder and the same
    # tables template; supports periodic dumps under "torchcomms_fr_trace".
    """Handler for TorchComms FlightRecorder trace data."""

    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns routes for the TorchComms trace and its JSON view.
        return [
            Route("/torchcomms_fr_trace", self._handle_torchcomms_fr_trace),
            Route("/torchcomms_fr_trace_json", self._handle_torchcomms_fr_trace_json),
        ]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the TorchComms FR nav link and its JSON link.
        return [
            NavLink("/torchcomms_fr_trace", "TorchComms FR"),
            NavLink("/torchcomms_fr_trace_json", "(JSON)"),
        ]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the shared FlightRecorder tables template.
        return {"fr_trace.html": FR_TRACE_TEMPLATE}

    def _handle_torchcomms_fr_trace(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches active-only TorchComms traces from all ranks and renders tables.
        addrs, resps = fetch_all(
            "torchcomms_fr_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return self._render_tables(req.frontend, addrs, list(resps))

    def _handle_torchcomms_fr_trace_json(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches active-only TorchComms traces from all ranks and renders pretty JSON.
        addrs, resps = fetch_all(
            "torchcomms_fr_trace_json", "onlyactive=true", timeout=self.fetch_timeout
        )
        return req.frontend.render_template(
            "json_resp.html",
            title="TorchComms FlightRecorder",
            addrs=addrs,
            resps=resps,
        )

    def _render_tables(
        self, server: FrontendServer, addrs: list[str], resps: list[Response]
    ) -> bytes:
        # [FUNCTION DOC] HELPER: Builds the database via the FlightRecorder builder and renders the
        # tables page titled for TorchComms with the fetch summary.
        db = FlightRecorderHandler._build_db(addrs, resps)
        return server.render_template(
            "fr_trace.html",
            title="TorchComms FlightRecorder",
            fetch_summary=format_fetch_summary(addrs, resps),
            groups=tabulate(db.groups, headers=Group._fields, tablefmt="html"),
            memberships=tabulate(
                db.memberships, headers=Membership._fields, tablefmt="html"
            ),
            collectives=tabulate(
                db.collectives, headers=Collective._fields, tablefmt="html"
            ),
            ncclcalls=tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="html"),
        )

    def dump(self) -> str | None:
        # [FUNCTION DOC] Returns plain-text tables from the TorchComms traces of all ranks, preceded by
        # a partial data summary if some ranks failed. Unlike the page handlers it does not
        # request active-only traces. Raises if all ranks failed.
        addrs, resps = fetch_all("torchcomms_fr_trace_json", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        db = FlightRecorderHandler._build_db(addrs, resps)
        parts.extend(
            [
                "=== TorchComms FR Trace ===",
                "--- Groups ---",
                tabulate(db.groups, headers=Group._fields, tablefmt="plain"),
                "--- Memberships ---",
                tabulate(db.memberships, headers=Membership._fields, tablefmt="plain"),
                "--- Collectives ---",
                tabulate(db.collectives, headers=Collective._fields, tablefmt="plain"),
                "--- NCCL Calls ---",
                tabulate(db.ncclcalls, headers=NCCLCall._fields, tablefmt="plain"),
            ]
        )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "torchcomms_fr_trace".
        return "torchcomms_fr_trace"


HEALTH_CHECK_TEMPLATE = """
{% extends "base.html" %}
{% block header %}
    <h1>{% block title %}TorchComms Health Check{% endblock %}</h1>
{% endblock %}
{% block content %}
    <h2>Health Status</h2>
    {% if fetch_summary %}<pre>{{ fetch_summary }}</pre>{% endif %}
    {% for i, (addr, resp) in enumerate(zip(addrs, resps)) %}
        <h3>Rank {{ i }}: {{ addr }}</h3>
        {% if resp.status_code != 200 %}
            <p>Failed to fetch: status={{ resp.status_code }}</p>
            <pre>{{ resp.text }}</pre>
        {% else %}
            <pre>{{ resp.text }}</pre>
        {% endif %}
    {% endfor %}
    {% if dump_triggered %}
        <h2>Flight Recorder Dump (triggered by unhealthy rank)</h2>
        {% for i, (addr, resp) in enumerate(zip(dump_addrs, dump_resps)) %}
            <h3>Rank {{ i }}: {{ addr }}</h3>
            {% if resp.status_code != 200 %}
                <p>Failed to dump: status={{ resp.status_code }}</p>
                <pre>{{ resp.text }}</pre>
            {% else %}
                <pre>{{ resp.text }}</pre>
            {% endif %}
        {% endfor %}
    {% endif %}
{% endblock %}
    """


class TorchCommsHealthCheckHandler(DebugHandler):
    # [CLASS DOC] Queries TorchComms health from all ranks; if any reachable rank reports itself
    # unhealthy, additionally requests a flight recorder dump file from all ranks. Used by the page
    # and by periodic dumps (name "torchcomms_health_check").
    """Health check that detects watchdog timeouts and triggers FR dumps."""

    def routes(self) -> list[Route]:
        # [FUNCTION DOC] Returns the route for the health check path.
        return [Route("/torchcomms_health_check", self._handle)]

    def nav_links(self) -> list[NavLink]:
        # [FUNCTION DOC] Returns the TorchComms Health nav link.
        return [NavLink("/torchcomms_health_check", "TorchComms Health")]

    def templates(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the health check page template.
        return {"torchcomms_health_check.html": HEALTH_CHECK_TEMPLATE}

    @staticmethod
    def _any_unhealthy(resps: list[Response]) -> bool:
        # [FUNCTION DOC] HELPER: True if any 200 response has JSON whose healthy field is false.
        # Non-200 responses and unparseable bodies (logged) are treated as healthy.
        """Return True if any rank explicitly reports itself as unhealthy.

        Only responses with a 200 status code are inspected.  Non-200
        responses (connection errors, timeouts, etc.) are intentionally
        ignored so that a single unreachable rank does not trigger an
        expensive Flight Recorder dump across the entire job.  A dump is
        only triggered when a rank is reachable *and* positively reports
        ``{"healthy": false}``.
        """
        for resp in resps:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    logger.exception(
                        "failed to parse health check response as JSON; "
                        "treating rank as healthy"
                    )
                    continue
                if not data.get("healthy", True):
                    return True
        return False

    def _handle(self, req: HTTPRequestHandler) -> bytes:
        # [FUNCTION DOC] HELPER: Fetches health from all ranks; if any is unhealthy also fetches the FR
        # dump file from all ranks. Renders the page with both sets of responses.
        addrs, resps = fetch_all("torchcomms_health_check", timeout=self.fetch_timeout)
        dump_triggered = False
        dump_addrs: list[str] = []
        dump_resps: list[Response] = []
        if self._any_unhealthy(resps):
            dump_addrs, dump_resps = fetch_all(
                "torchcomms_fr_dump_file", timeout=self.fetch_timeout
            )
            dump_triggered = True
        return req.frontend.render_template(
            "torchcomms_health_check.html",
            fetch_summary=format_fetch_summary(addrs, resps),
            addrs=addrs,
            resps=resps,
            dump_triggered=dump_triggered,
            dump_addrs=dump_addrs,
            dump_resps=dump_resps,
        )

    def dump(self) -> str | None:
        # [FUNCTION DOC] Returns text with an optional partial data summary, one section per rank health
        # response, and, if any rank is unhealthy, a section with each rank FR dump file result.
        addrs, resps = fetch_all("torchcomms_health_check", timeout=self.fetch_timeout)
        parts: list[str] = []
        summary = format_fetch_summary(addrs, resps)
        if summary:
            parts.append(summary)
            parts.append("")
        for i, (addr, resp) in enumerate(zip(addrs, resps)):
            parts.append(f"=== Rank {i}: {addr} ===")
            parts.append(
                resp.text if resp.status_code == 200 else f"Error: {resp.status_code}"
            )
        if self._any_unhealthy(resps):
            parts.append("")
            parts.append("=== Unhealthy rank detected, triggering FR dump ===")
            dump_addrs, dump_resps = fetch_all(
                "torchcomms_fr_dump_file", timeout=self.fetch_timeout
            )
            for i, (addr, resp) in enumerate(zip(dump_addrs, dump_resps)):
                parts.append(f"--- Rank {i}: {addr} ---")
                parts.append(
                    resp.text
                    if resp.status_code == 200
                    else f"Error: {resp.status_code}"
                )
        return "\n".join(parts)

    def dump_filename(self) -> str:
        # [FUNCTION DOC] Returns the fixed dump file name prefix "torchcomms_health_check".
        return "torchcomms_health_check"


def default_handlers() -> list[DebugHandler]:
    # [FUNCTION DOC] Returns fresh instances of the default handlers: index, stacks, py-spy,
    # FlightRecorder, TorchComms FlightRecorder, TorchComms health check, profiler, wait counters
    # and TCPStore.
    return [
        IndexHandler(),
        StacksHandler(),
        PySpyHandler(),
        FlightRecorderHandler(),
        TorchCommsFlightRecorderHandler(),
        TorchCommsHealthCheckHandler(),
        ProfilerHandler(),
        WaitCountersHandler(),
        TCPStoreHandler(),
    ]
