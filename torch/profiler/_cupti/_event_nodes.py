# mypy: allow-untyped-defs
# [MODULE DOC]
# Maps CUPTI CUDA_EVENT activity records back to the event-record nodes of the CUDA graph
# that produced them. Contains the process-wide _EventNodeRecorder singleton (records each
# graph's event-record node ids in node-id order at graph instantiate), the pure
# resolve_window function (positional per-launch join of records to nodes), and a test reset
# helper. The join is by position within a launch, never by event id. Import-time side
# effects: none beyond defining the class and a tuple of node types that are refused.
"""Passive bridge: map a CUDA_EVENT record back to the CUDA-graph event-record node.

CUPTI emits a ``CUDA_EVENT`` activity record for each event-record node executed in a
graph (e.g. the nodes NCCL inserts under ``NCCL_GRAPH_MIXING_SUPPORT=1``), but the record
carries only an ``event_id`` (keyed to the CUDA event object) -- never a ``graph_node_id``,
so it cannot be joined to the graph node the way kernel/memcpy/memset records are. There is
also no API to query a CUevent's ``event_id``, and the events are owned by whoever built the
graph (NCCL), so eagerly re-recording them to learn a mapping is unsafe.

The join is instead POSITIONAL, per launch, from records that already flow:

  * A graph launch's graphed kernel/memset records share the launch's ``correlation_id`` and
    carry the exec graph id in the upper 32 bits of their ``graph_node_id``. So per launch,
    ``correlation_id -> exec_graph_id``.
  * The same launch's ``CUDA_EVENT`` records share that ``correlation_id`` and carry an
    increasing ``cuda_event_sync_id`` that follows NODE-ID order.
  * :meth:`_EventNodeRecorder.arm` records, at graph instantiate, each graph's ``event_record``
    node ids in node-id order (from ``get_graph_data()``).

The invariant the join rests on:

    The k-th ``CUDA_EVENT`` record of a launch, by sync id, is the k-th ``event_record``
    node of that launch's graph, by node id.

Node ids are assigned in node-CREATION order, which is also the order ``get_graph_data()``
returns nodes in, so no sorting is needed. Note this is deliberately NOT execution order:
on a graph with concurrent branches the branches' event nodes execute in an order that
varies run to run, while ``cuda_event_sync_id`` stays fixed to node-id order. Ordering by
anything derived from the DAG (dependency depth, topological rank) therefore mislabels
nodes on any ordinary fork/join graph.

Resolution keys on POSITION, never on ``event_id``: a producer like NCCL/FSDP recycles one
``cudaEvent_t`` across many event-record nodes, so ``event_id`` is stable but not unique to
a node -- keying on it would collapse every recycled record onto a single node. A launch is
resolved only when its event-record count matches the graph's node count; otherwise its
records resolve to ``None``, never guessed.

(If CUPTI ever adds ``graphNodeId`` to ``CUpti_ActivityCudaEvent`` this whole module goes
away -- the record would join like any other graphed activity.)
"""

from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Self


# Node kinds whose bodies get_graph_data() does not descend into. Their event nodes are
# invisible here, and at replay CUPTI reports nested nodes under the parent exec graph id
# with ids appended after the top-level ones -- so a nested event node would sort last by
# node id regardless of when it ran, and a conditional body executes a variable number of
# times per launch. Refuse to resolve against such a graph rather than mis-assign.
_OPAQUE_BODY_NODE_TYPES = ("child_graph", "conditional")


class _EventNodeRecorder:
    # [CLASS DOC] HELPER:
    # Process-wide singleton (construction returns the one shared instance, guarded by a
    # class-level lock). State: graph_event_nodes maps an exec graph id to the list of
    # event-record node ids (tools ids) of that graph in node-id order; a private handle to
    # the instantiate hook registration (none until armed). Invariant: only graphs without
    # child-graph or conditional nodes and with at least one event node are present.
    # Key methods: arm, purge_exec_ids.
    """Process-global record of each CUDA graph's event-record nodes, in node-id order.

    Process-wide singleton armed once via :meth:`arm` (before graphs are captured), like
    the graph-dependency recorder. :attr:`graph_event_nodes` maps ``exec_graph_id ->
    event-node tools_ids`` in node-id order. Resolution is positional against this map (see
    :func:`resolve_window`); there is no learned ``event_id`` map.
    """

    _instance: _EventNodeRecorder | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> Self:
        # [FUNCTION DOC] HELPER:
        # Singleton constructor. Under the class lock, creates the instance on first call
        # (initializing its state) and stores it on the class; always returns that one
        # instance. Thread-safe. Takes no arguments.
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
            return cls._instance

    def _init(self) -> None:
        # [FUNCTION DOC] HELPER:
        # Sets up an empty graph-to-event-node mapping and an unset hook handle. Called once
        # by the singleton constructor.
        self.graph_event_nodes: dict[int, list[int]] = {}
        self._handle: Any = None

    def arm(self) -> None:
        # [FUNCTION DOC]
        # Registers the graph-instantiate hook (from torch.cuda.graphs) so every later graph
        # instantiate is recorded. Idempotent: does nothing if already armed. Side effect:
        # imports torch.cuda.graphs lazily and stores the hook handle. Graphs instantiated
        # before arming are never recorded (no backfill).
        """Register the graph-instantiate hook once (idempotent).

        Must be called BEFORE the graphs of interest are captured: the hook fires at
        instantiate, so a graph already instantiated when this runs is never recorded and
        arming does not backfill it. Records from such a graph's launches then resolve to
        ``None`` (see :func:`resolve_window`, which refuses to guess) rather than failing.
        """
        if self._handle is not None:
            return
        from torch.cuda.graphs import register_graph_instantiate_hook

        self._handle = register_graph_instantiate_hook(self._on_instantiate)

    def _on_instantiate(self, torch_cuda_graph: Any) -> None:
        # [FUNCTION DOC] HELPER:
        # Instantiate hook. Reads the graph's node data; silently returns on runtime,
        # attribute or key errors (bindings or driver unavailable), if any node is a child
        # graph or conditional node, or if there are no event-record nodes. Otherwise stores
        # the event-record node ids (in node-id order) under the exec graph id and adds that
        # id to the graph object's set of recorded exec ids so the destroy hook can purge it.
        # Template is live here (both keep_graph modes), so get_graph_data() works. Degrade to
        # no record when cuda.bindings / a recent driver is unavailable.
        try:
            data = torch_cuda_graph.get_graph_data()
            nodes = data["nodes"]
            exec_graph_id = data["exec_graph_id"]
        except (RuntimeError, AttributeError, KeyError):
            return
        if any(n["node_type"] in _OPAQUE_BODY_NODE_TYPES for n in nodes):
            return
        # get_graph_data() returns nodes in node-id order, which is node-creation order, so
        # filtering preserves it -- no sort.
        ordered = [n["tools_id"] for n in nodes if n["node_type"] == "event_record"]
        if not ordered:  # no event nodes -> nothing to track
            return
        # Track the exec id so the destroy hook can purge this entry.
        self.graph_event_nodes[exec_graph_id] = ordered
        torch_cuda_graph._recorded_exec_ids.add(exec_graph_id)

    def purge_exec_ids(self, exec_ids: set[int]) -> None:
        # [FUNCTION DOC]
        # Removes the recorded event-node lists for each given exec graph id; ids that are
        # not present are ignored. Mutates the recorder's mapping; returns nothing.
        """Drop recorded state for destroyed graphs (called from the graph-destroy hook)."""
        for eid in exec_ids:
            self.graph_event_nodes.pop(eid, None)


def resolve_window(
    recorder: _EventNodeRecorder,
    corr_exec_pairs: Any,
    event_rows: list[tuple[int, int]],
) -> list[int | None]:
    # [FUNCTION DOC]
    # Resolves each CUDA_EVENT record of one export window to a graph node id. Inputs: the
    # recorder, an iterable of (correlation id, graph node id) pairs for graphed work records
    # (the exec graph id is the upper 32 bits of the node id; zero node ids are skipped), and
    # a list of (correlation id, sync id) rows. Groups rows by correlation id (one launch),
    # sorts each group by sync id, and assigns the k-th row the k-th recorded event node of
    # that launch's exec graph. Returns a list aligned to the rows with None for any launch
    # whose exec graph is unknown/unrecorded or whose row count differs from the node count.
    # Pure apart from reading the recorder; raises nothing itself.
    """Resolve one export window's CUDA_EVENT records to graph event-record nodes.

    ``corr_exec_pairs`` is ``(correlation_id, graph_node_id)`` for the window's graphed work
    records (kernels/memsets); ``event_rows`` is ``(correlation_id, cuda_event_sync_id)`` per
    CUDA_EVENT record. Returns the resolved ``graph_node_id`` (or ``None``) aligned to
    ``event_rows``. Pure: the numpy marshalling stays in the observer.

    Resolution is positional per launch: within a launch (one ``correlation_id``) the records
    are sorted by sync id and the k-th record is assigned the k-th event node of that launch's
    exec graph, by node id. A launch resolves only when its record count matches the graph's
    node count; that count check is the safety net against nested bodies, dropped records, and
    launches split across export windows. This never keys on the CUDA event object, so it is
    correct when a producer recycles one ``cudaEvent_t`` across nodes.
    """
    corr_to_exec: dict[int, int] = {}
    for corr, gnid in corr_exec_pairs:
        if gnid:
            corr_to_exec.setdefault(corr, gnid >> 32)
    by_corr: dict[int, list[tuple[int, int]]] = {}
    for i, (corr, sid) in enumerate(event_rows):
        by_corr.setdefault(corr, []).append((sid, i))
    resolved: list[int | None] = [None] * len(event_rows)
    for corr, rows in by_corr.items():
        exec_id = corr_to_exec.get(corr)
        if exec_id is None:
            continue
        ordered = recorder.graph_event_nodes.get(exec_id)
        if not ordered or len(ordered) != len(rows):
            continue
        rows.sort()  # by sync id, which follows node-id order within the launch
        for pos, (_sid, i) in enumerate(rows):
            resolved[i] = ordered[pos]
    return resolved


def _reset_for_test() -> None:
    # [FUNCTION DOC] HELPER:
    # Test-only reset. Under the singleton lock, removes the instantiate hook if one was
    # registered and clears the stored singleton so the next construction starts fresh.
    """Test-only: unregister the instantiate hook and drop the singleton, so the next
    ``_EventNodeRecorder()`` starts unarmed with an empty map."""
    with _EventNodeRecorder._instance_lock:
        inst = _EventNodeRecorder._instance
        if inst is not None and inst._handle is not None:
            inst._handle.remove()
        _EventNodeRecorder._instance = None
