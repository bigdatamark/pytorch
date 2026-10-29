# mypy: allow-untyped-defs
# [MODULE DOC]
# Provides a process-wide recorder of CUDA-graph topology for the CUPTI profiler: per-node
# predecessor lists, event handles of event-record nodes, and host-function names/addresses
# of host nodes, all keyed by graph node id. It is armed once early (before graphs are
# instantiated) via a graph-instantiate hook, and its maps persist for the whole process so
# per-window observers can read topology recorded before they existed. Import-time side
# effects: none beyond defining the class and a test reset helper.
"""Process-global CUDA-graph node->node dependency recorder for the CUPTI profiler.

The topology of a CUDA graph is only readable (via ``get_graph_data()``) while its
template is live, which -- for ``keep_graph=False`` graphs -- is only during the
``register_graph_instantiate_hook`` callback fired inside ``CUDAGraph.instantiate()``.
A workload typically captures/instantiates its graphs ONCE (during warm-up) and then
only replays them, so the recording hook must already be armed before that one-time
capture. The per-window ``ProfilerObserver`` cannot do this: it is created (and its
hook registered) at ``prepare_trace`` -- long after warm-up -- and torn down between
windows, so its hook never observes an ``instantiate()`` and its map stays empty.

This recorder decouples recording from the observer lifecycle. It is armed once, early
(from ``torch.profiler`` when ``enable_graph_dependencies`` is set, i.e. at profiler
construction, before the training loop), and its map persists for the process. Each
observer shares this map by reference, so its dependency resolver reads topology
recorded before that observer existed. ``keep_graph`` is irrelevant: the hook runs
during ``instantiate()`` while the template is still live.
"""

from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Self


class _GraphDependencyRecorder:
    # [CLASS DOC] HELPER:
    # Process-wide singleton (construction returns the one shared instance, guarded by a
    # class-level lock). State: deps (node id to predecessor node ids), event_record_events
    # (node id to event handle for event-record nodes), host_fns (node id to name and address
    # for host nodes), and a private hook handle (unset until armed). The maps are shared by
    # reference with observers and only grow, except for purging done elsewhere by exec id.
    # Key method: arm.
    """Persistent recorder of graph_node_id -> predecessor graph_node_ids edges.

    Process-wide singleton, like ``CuptiMonitor``: ``_GraphDependencyRecorder()`` returns
    the one instance, constructed on first call. Edges are keyed by ``tools_id`` (== the
    CUPTI ``graph_node_id`` that joins to profiler kernel records). Armed at most once via
    :meth:`arm`; :attr:`deps` is the shared map read (by reference) by every observer's
    dependency resolver.
    """

    _instance: _GraphDependencyRecorder | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> Self:
        # [FUNCTION DOC] HELPER:
        # Singleton constructor. Under the class lock, creates and initializes the instance
        # on first call, stores it on the class, and always returns that one instance.
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
            return cls._instance

    def _init(self) -> None:
        # [FUNCTION DOC] HELPER:
        # Creates the three empty maps (dependencies, event handles, host functions) and an
        # unset hook handle. Called once by the singleton constructor.
        self.deps: dict[int, list[int]] = {}
        # graph_node_id -> cudaEvent_t handle for event-record nodes. The CUPTI record for the
        # node (a CUDA_EVENT activity) carries no event handle, so this is the only way to tag
        # the rendered EventRecord span with the event it records -- which, together with the
        # node->node dependency arrows, shows what downstream nodes wait on (see get_graph_data).
        self.event_record_events: dict[int, int] = {}
        # graph_node_id -> (host_fn_name, host_fn_addr) for host nodes, recorded in the same
        # instantiate pass (host nodes carry no name in the CUPTI record; see get_graph_data).
        self.host_fns: dict[int, tuple[str | None, int]] = {}
        self._handle: Any = None

    def arm(self) -> None:
        # [FUNCTION DOC]
        # Registers the graph-instantiate hook from torch.cuda.graphs so later graph
        # instantiations are recorded. Idempotent. Side effect: lazy import and storing the
        # hook handle. Graphs instantiated before arming are not recorded.
        """Register the graph-instantiate hook once (idempotent)."""
        if self._handle is not None:
            return
        from torch.cuda.graphs import register_graph_instantiate_hook

        self._handle = register_graph_instantiate_hook(self._on_instantiate)

    def _on_instantiate(self, torch_cuda_graph: Any) -> None:
        # [FUNCTION DOC] HELPER:
        # Instantiate hook. Reads the graph's node data and silently returns on runtime,
        # attribute or key errors. Builds three maps from the nodes: node id to predecessor
        # node ids (only nodes having dependencies), node id to event handle (event-record
        # nodes with a nonzero handle), and node id to host function name and address (host
        # nodes). Returns without change if all three are empty; otherwise merges them into
        # the recorder and adds the exec graph id (upper 32 bits of any recorded node id) to
        # the graph object's set of recorded exec ids so the destroy hook can purge entries.
        # This class has no purge method; removal is done by an external destroy hook.
        # We hold the live graph here, so get_graph_data() works during instantiate()
        # for both keep_graph modes (the template is destroyed only afterwards). Raises
        # when cuda.bindings / a recent driver is unavailable -- degrade to no records.
        try:
            nodes = torch_cuda_graph.get_graph_data()["nodes"]
        except (RuntimeError, AttributeError, KeyError):
            return
        recorded = {
            n["tools_id"]: [nodes[i]["tools_id"] for i in n["dependencies"]]
            for n in nodes
            if n["dependencies"]
        }
        events = {
            n["tools_id"]: n["event_ptr"]
            for n in nodes
            if n["node_type"] == "event_record" and n["event_ptr"]
        }
        host = {
            n["tools_id"]: (n["host_fn_name"], n["host_fn_addr"])
            for n in nodes
            if n["node_type"] == "host"
        }
        if not recorded and not events and not host:
            return
        self.deps.update(recorded)
        self.event_record_events.update(events)
        self.host_fns.update(host)
        # Track the exec graph id (tools_id >> 32, shared by all the graph's nodes) so the
        # observer's graph-destroy hook can purge this graph's entries on destruction.
        any_id = (
            next(iter(recorded))
            if recorded
            else next(iter(events))
            if events
            else next(iter(host))
        )
        torch_cuda_graph._recorded_exec_ids.add(any_id >> 32)


def _reset_for_test() -> None:
    # [FUNCTION DOC] HELPER:
    # Test-only reset. Under the singleton lock, removes the instantiate hook if registered
    # and clears the stored singleton so the next construction starts unarmed and empty.
    """Test-only: unregister the instantiate hook and drop the singleton, so the next
    ``_GraphDependencyRecorder()`` starts unarmed with empty maps."""
    with _GraphDependencyRecorder._instance_lock:
        inst = _GraphDependencyRecorder._instance
        if inst is not None and inst._handle is not None:
            inst._handle.remove()
        _GraphDependencyRecorder._instance = None
