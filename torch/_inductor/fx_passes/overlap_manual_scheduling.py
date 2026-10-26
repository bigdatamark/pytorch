# [MODULE DOC] Manual overlap scheduling for collectives: given user-specified module FQN plans, buckets the all-gathers,
# reduce-scatters and all-reduces inside each module into single fused collectives, then reorders the fx graph (and optionally adds
# control dependencies) so all-gathers prefetch during the previous module's compute and reduce-scatter waits are deferred.
# Main contents: ManualOverlapPreservingBucketer, ManualOverlapScheduler and the entry point manual_overlap_bucketing.
# Import-time side effect: creates the module logger only.
from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    _get_collective_node_from_wait,
    _schedulable_wait_node,
    BucketMode,
    has_mergeable_all_gather_convert_dtype,
    is_all_gather_into_tensor as is_all_gather,
    is_all_reduce_tensor,
    is_fsdp_all_gather,
    is_fsdp_reduce_scatter,
    is_reduce_scatter_tensor as is_reduce_scatter,
    merge_all_gather_bucket,
    merge_all_reduce_bucket,
    merge_reduce_scatter_bucket,
)
from torch._inductor.fx_passes.overlap_preserving_bucketer import (
    get_full_bucket_key,
    OverlapPreservingBucketer,
)
from torch._inductor.fx_passes.overlap_scheduling import (
    CollectiveInfo,
    is_compute_node,
    OverlapScheduler,
)
from torch.utils._ordered_set import OrderedSet

from .graph_view import get_subgraph_by_path, GraphView, make_graph_view


if TYPE_CHECKING:
    from collections.abc import Callable

import logging


logger = logging.getLogger(__name__)


def _collect_nodes_must_be_after(node: fx.Node) -> list[fx.Node]:
    # [FUNCTION DOC] HELPER: Breadth-first forward walk from node collecting it and each transitive user whose inputs are all already in the
    # collected set (so nodes depending on anything outside are excluded). Returns the list in discovery order, starting with node.
    """BFS forward collecting node and its transitive users with no external inputs."""
    result: list[fx.Node] = [node]
    result_set: OrderedSet[fx.Node] = OrderedSet([node])
    i = 0
    while i < len(result):
        for user in result[i].users:
            if user not in result_set and all(
                inp in result_set for inp in user.all_input_nodes
            ):
                result_set.add(user)
                result.append(user)
        i += 1
    return result


def _collect_nodes_must_be_before(
    node: fx.Node, node_positions: dict[fx.Node, int]
) -> list[fx.Node]:
    # [FUNCTION DOC] HELPER: Walks backward from node through input nodes (placeholders excluded) and returns the visited nodes sorted by
    # their graph position from the given position map (topological order). Includes node itself.
    """BFS backward collecting node and its non-placeholder dependencies, topo-sorted."""
    visited: OrderedSet[fx.Node] = OrderedSet()
    queue = [node]
    while queue:
        cur = queue.pop()
        if cur in visited or cur.op == "placeholder":
            continue
        visited.add(cur)
        queue.extend(cur.all_input_nodes)
    return sorted(visited, key=lambda n: node_positions[n])


def _bucket_trace_inputs(
    coll_node: fx.Node, node_in: object, group_name_arg: int
) -> list[fx.Node]:
    # [FUNCTION DOC] HELPER: Returns the nodes that anchor insertion of a bucketed collective: the collective's input node and, if the group name
    # argument at the given position is an fx node, that node too. Raises an error if the input is not an fx node.
    if not isinstance(node_in, fx.Node):
        raise AssertionError(f"expected node input to be a Node, got {type(node_in)}")
    inputs = [node_in]

    group_name = coll_node.args[group_name_arg]
    if isinstance(group_name, fx.Node):
        inputs.append(group_name)
    return inputs


def _all_gather_bucket_trace_inputs(coll_node: fx.Node) -> list[fx.Node]:
    # [FUNCTION DOC] HELPER: Bucket-trace inputs for an all-gather start node. If the input is a mergeable dtype conversion, anchors on the
    # converted source tensor instead (the conversion is erased by the bucket). Group name is argument 2.
    # Raises an error if the expected input is not an fx node.
    node_in: object = coll_node.args[0]
    # The dtype conversion is erased by the all-gather bucket trace, so anchor
    # insertion on the tensor that remains as a graph input to the bucket.
    if has_mergeable_all_gather_convert_dtype(coll_node):
        if not isinstance(node_in, fx.Node):
            raise AssertionError(
                f"expected node input to be a Node, got {type(node_in)}"
            )
        node_in = node_in.args[0]
    return _bucket_trace_inputs(coll_node, node_in, group_name_arg=2)


def _reduce_scatter_bucket_trace_inputs(coll_node: fx.Node) -> list[fx.Node]:
    # [FUNCTION DOC] HELPER: Bucket-trace inputs for a reduce-scatter start node (tensor is argument 0, group name argument 3).
    return _bucket_trace_inputs(coll_node, coll_node.args[0], group_name_arg=3)


def _all_reduce_bucket_trace_inputs(coll_node: fx.Node) -> list[fx.Node]:
    # [FUNCTION DOC] HELPER: Bucket-trace inputs for an all-reduce start node (tensor is argument 0, group name argument 2).
    # all_reduce(tensor, reduce_op, group_name): tensor at 0, group_name at 2.
    return _bucket_trace_inputs(coll_node, coll_node.args[0], group_name_arg=2)


def _move_wait_users_after_latest_inputs(
    graph: fx.Graph,
    replacements: dict[fx.Node, fx.Node],
    replaced_users: dict[fx.Node, list[fx.Node]],
) -> None:
    # [FUNCTION DOC] HELPER: After old waits are replaced by bucketed outputs, moves existing consumers of the old waits that now sit before
    # the new output so they come after their latest input. Uses a work queue: each moved node's users are re-queued, and positions are
    # recomputed after every move. Args: graph, mapping of old to new outputs, and the saved users of each old wait.
    # Side effect: reorders nodes in the graph in place. Returns nothing.
    node_positions = {n: i for i, n in enumerate(graph.nodes)}
    initial_users: OrderedSet[fx.Node] = OrderedSet()
    for old_out, new_out in replacements.items():
        if new_out not in node_positions:
            continue
        for user in replaced_users.get(old_out, []):
            if (
                user in node_positions
                and user.op != "output"
                and node_positions[user] < node_positions[new_out]
            ):
                initial_users.add(user)

    pending = sorted(initial_users, key=lambda n: node_positions[n])
    queued = OrderedSet(pending)
    while pending:
        node = pending.pop(0)
        queued.discard(node)

        node_positions = {n: i for i, n in enumerate(graph.nodes)}
        if node not in node_positions:
            continue

        input_nodes = [inp for inp in node.all_input_nodes if inp in node_positions]
        if not input_nodes:
            continue

        latest_input = max(input_nodes, key=lambda n: node_positions[n])
        if node_positions[node] >= node_positions[latest_input]:
            continue

        # Replacing old waits can leave existing consumers before the new bucket
        # outputs. Pull each affected consumer after its latest input.
        latest_input.append(node)
        node_positions = {n: i for i, n in enumerate(graph.nodes)}
        for user in node.users:
            if user in node_positions and user.op != "output" and user not in queued:
                queued.add(user)
                pending.append(user)


def _move_overlap_nodes(
    graph: fx.Graph,
    overlap_deps: dict[fx.Node, OrderedSet[fx.Node]],
    bucketed_node_types: dict[fx.Node, str],
) -> None:
    # [FUNCTION DOC] HELPER: Applies overlap dependencies by physically moving nodes. Does nothing if there are no dependencies.
    # For targets depending on bucketed reduce-scatter or all-reduce starts, moves the target and its must-follow nodes to just after the
    # latest such start. For targets depending on bucketed all-gather starts located at or after the target, moves each start and its
    # dependencies to just before the target. Args: graph, mapping target to its set of source nodes, mapping node to type string.
    # Side effect: reorders the graph in place.
    if not overlap_deps:
        return

    rs_defer: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    ag_prefetch: dict[fx.Node, list[fx.Node]] = defaultdict(list)

    for target, sources in overlap_deps.items():
        for source in sources:
            source_type = bucketed_node_types.get(source, "")
            if source_type.startswith(
                ("bucketed_reduce_scatter", "bucketed_all_reduce")
            ):
                rs_defer[target].append(source)
            elif source_type.startswith("bucketed_all_gather"):
                ag_prefetch[target].append(source)

    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    for rs_wait, rs_starts in rs_defer.items():
        latest_rs_start = max(rs_starts, key=lambda n: node_positions[n])
        node_insert_after = latest_rs_start
        for node in _collect_nodes_must_be_after(rs_wait):
            node_insert_after.append(node)
            node_insert_after = node

    # Recompute positions after RS moves
    node_positions = {n: i for i, n in enumerate(graph.nodes)}

    for ag_wait, ag_prefetch_starts in ag_prefetch.items():
        ag_wait_pos = node_positions[ag_wait]
        sorted_starts = sorted(ag_prefetch_starts, key=lambda n: node_positions[n])
        for ag_start in sorted_starts:
            if node_positions[ag_start] < ag_wait_pos:
                continue
            for node in _collect_nodes_must_be_before(ag_start, node_positions):
                ag_wait.prepend(node)


class ManualOverlapPreservingBucketer(OverlapPreservingBucketer):
    # [CLASS DOC] Overlap-preserving bucketer that buckets collectives chosen by the user (per module) rather than by a heuristic.
    # State: node_to_wait_map maps each new bucketed wait to the canonical (last) new wait; bucketed_node_types maps new nodes to a type
    # string (bucketed_all_gather, bucketed_reduce_scatter, bucketed_all_reduce, plus a wait suffix), scoped per instance.
    # Key methods: manual_bucket_collectives (groups and buckets) and the private bucket/split helpers.
    """
    Buckets collective operations based on user specifications.
    The actual bucket happens in bucket_collectives, where all-gathers/reduce-scatters in
        `nodes` will be buckted one single all-gather/reduce-scatter.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        # [FUNCTION DOC] Forwards all arguments to the base bucketer and initializes the empty wait map and bucketed node type map.
        super().__init__(*args, **kwargs)
        self.node_to_wait_map: dict[fx.Node, fx.Node] = defaultdict()
        # Maps bucketed nodes to their type string, scoped to this bucketer
        # instance so metadata doesn't leak across separate invocations.
        self.bucketed_node_types: dict[fx.Node, str] = {}

    def _bucket_group(self, coll_nodes: list[fx.Node]) -> None:
        # [FUNCTION DOC] HELPER: Fuses the given same-kind collective start nodes into one bucketed collective. Chooses the merge function and
        # type string by whether the first node is an all-gather, reduce-scatter or all-reduce; raises ValueError for any other kind and an error
        # if the list is empty or no new wait is produced. Inserts the new collective after its latest input and the new wait no earlier than the
        # first old wait, repositions consumers of old waits, then records the types of the new start and wait nodes and maps each new wait to the
        # last new wait. Side effects: mutates the graph and the bucketer's maps, logs at debug level.
        if len(coll_nodes) <= 0:
            raise AssertionError("bucketed coll_nodes should have nonzero node")

        # Graph order changes after each bucket, so positions must be fresh.
        node_positions = {n: i for i, n in enumerate(self.graph.nodes)}
        waits = [self.collective_info[n].wait_node for n in coll_nodes]
        first_wait = min(waits, key=lambda w: node_positions[w])
        first = min(coll_nodes, key=lambda n: node_positions[n])
        replaced_users = {wait: list(wait.users) for wait in waits}

        if is_all_gather(first):
            bucket_trace_inputs = _all_gather_bucket_trace_inputs
            merge_bucket = merge_all_gather_bucket
            node_type = "bucketed_all_gather"
        elif is_reduce_scatter(first):
            bucket_trace_inputs = _reduce_scatter_bucket_trace_inputs
            merge_bucket = merge_reduce_scatter_bucket
            node_type = "bucketed_reduce_scatter"
        elif is_all_reduce_tensor(first):
            bucket_trace_inputs = _all_reduce_bucket_trace_inputs
            merge_bucket = merge_all_reduce_bucket
            node_type = "bucketed_all_reduce"
        else:
            raise ValueError(
                "bucket non all_gather/reduce_scatter/all_reduce node is not supported"
            )

        # coll_nodes order is used for tensor packing and may differ from
        # graph order. Insert the bucketed collective after its latest input.
        bucket_inputs = [inp for n in coll_nodes for inp in bucket_trace_inputs(n)]
        anchor = max([first, *bucket_inputs], key=lambda n: node_positions[n])
        next_node = anchor.next
        coll_node_set = OrderedSet(coll_nodes)
        while next_node in coll_node_set:
            next_node = next_node.next
        # Use the earliest old wait unless it precedes the bucket insertion
        # point; otherwise keep wait/output nodes with the traced bucket.
        wait_insertion_point = max(
            (first_wait, next_node), key=lambda n: node_positions[n]
        )

        new_nodes, replacements = merge_bucket(
            self.graph,
            coll_nodes,
            wait_insertion_point=wait_insertion_point,
            insert_before=next_node,
            mode=self.bucket_mode,
        )
        _move_wait_users_after_latest_inputs(self.graph, replacements, replaced_users)

        logger.debug(f"bucketing nodes: {coll_nodes} into {new_nodes}")  # noqa: G004

        # Identify the new wait(s) and their collective start in a single pass
        wait_to_start = {
            n: start
            for n in new_nodes
            if (start := _get_collective_node_from_wait(n)) is not None
        }
        if len(wait_to_start) < 1:
            raise AssertionError(
                f"Expected at least one new wait, got none in {new_nodes}"
            )
        new_waits = list(wait_to_start)
        new_start: fx.Node = wait_to_start[new_waits[0]]
        # Use last wait as the canonical wait for scheduling (same node when len == 1)
        new_wait = new_waits[-1]

        # Track bucketed node types on this bucketer instance so it doesn't leak
        # when the same graph is processed by multiple ManualOverlapScheduler
        # invocations (e.g. separate forward and backward passes).
        wait_set = OrderedSet(new_waits)
        for n in new_nodes:
            if n in wait_set:
                self.bucketed_node_types[n] = node_type + "_wait"
                self.node_to_wait_map[n] = new_wait
            elif n is new_start:
                self.bucketed_node_types[n] = node_type

    def _split_independent_collectives(
        self, coll_nodes: OrderedSet[fx.Node], scope_nodes: list[fx.Node]
    ) -> list[list[fx.Node]]:
        # [FUNCTION DOC] HELPER: Partitions same-key collectives into dependency-free buckets so that no fused collective's input depends on its own
        # output. Computes for each collective the length of the longest chain of same-key collectives ending at it using one forward pass over the
        # scope nodes in original order (a collective's chain propagates through its wait node), then groups by that level. Returns a list of
        # collective lists ordered by level. Args: set of start nodes and the scope nodes. Assumes each collective has recorded info.
        """Partition same-key collectives so no bucket contains a collective that
        depends on another member's result.

        Bucketing fuses collectives into one op: inputs are concatenated before
        the fused collective and outputs split after. If collective B's input
        depends on collective A's output (A's wait), fusing A and B makes the
        merged collective's input depend on its own output -- a graph cycle that
        later fails region topological sort. This arises e.g. in loss-parallel
        cross-entropy, which emits a sumexp and a result sum-all_reduce on the
        same (group, reduce_op, dtype) key where result depends on sumexp.

        Each collective is placed in the bucket equal to the longest chain of
        same-key collectives ending at it (its Mirsky level). Collectives at the
        same level are mutually independent, so grouping by level gives the
        minimum number of dependency-free buckets. Levels are computed with a
        single topological forward pass over ``scope_nodes`` that propagates,
        along real graph edges, the deepest same-key chain reaching each node --
        O(len(scope_nodes)) rather than O(len(coll_nodes)^2) pairwise ancestor
        queries. This is exact because same-key collectives in one bucketing
        scope depend on each other only through in-scope nodes.
        """
        wait_to_start = {self.collective_info[c].wait_node: c for c in coll_nodes}

        # ``reach[node]`` = deepest same-key collective chain in node's cone;
        # ``level[c]`` = that chain length including c (its bucket index + 1).
        reach: dict[fx.Node, int] = {}
        level: dict[fx.Node, int] = {}
        for node in sorted(scope_nodes, key=lambda n: self.node_idx[n]):
            incoming = max(
                (reach.get(inp, 0) for inp in node.all_input_nodes), default=0
            )
            node_reach = incoming
            if node in coll_nodes:
                level[node] = incoming + 1
            # A collective's chain becomes reachable through its wait (its result).
            start = wait_to_start.get(node)
            if start is not None:
                node_reach = max(node_reach, level[start])
            reach[node] = node_reach

        buckets: dict[int, list[fx.Node]] = defaultdict(list)
        for c in coll_nodes:
            buckets[level[c]].append(c)
        return [buckets[lvl] for lvl in sorted(buckets)]

    def manual_bucket_collectives(self, nodes: list[fx.Node]) -> None:
        # [FUNCTION DOC] Buckets the fsdp all-gathers, fsdp reduce-scatters and all-reduces among the given nodes. Ignores nodes that are not known
        # collectives, groups the rest by their full bucket key (custom ops mode; nodes without a key are dropped), splits each group into
        # dependency-free sub-buckets, and buckets each one. Side effect: mutates the graph. Returns nothing.
        # KNOWN ISSUE: the docstring mentions only all-gather and reduce-scatter but all-reduce is also bucketed.
        """
        Bucket all all-gather/reduce-scatter nodes from nodes into one all-gather/reduce-scatter.
        """
        # Filter out valid collectives
        collectives = [n for n in nodes if n in self.collective_info]
        if collectives == []:
            return
        grouped_collectives: dict[object, OrderedSet[fx.Node]] = defaultdict(OrderedSet)
        for node in collectives:
            if not (
                is_fsdp_all_gather(node, self.node_ancestors)
                or is_fsdp_reduce_scatter(node)
                or is_all_reduce_tensor(node)
            ):
                continue
            key = get_full_bucket_key(node, "custom_ops")
            if key is not None:
                grouped_collectives[key].add(node)

        # Split each key-group into dependency-free sub-buckets so fusing never
        # makes a bucketed collective's input depend on its own output.
        sub_buckets: list[list[fx.Node]] = []
        for key, key_nodes in grouped_collectives.items():
            sub_buckets.extend(self._split_independent_collectives(key_nodes, nodes))

        for sub_bucket in sub_buckets:
            self._bucket_group(sub_bucket)


class ManualOverlapScheduler(OverlapScheduler):
    # [CLASS DOC] Overlap scheduler that buckets and reorders collectives following user module plans instead of runtime estimates.
    # State: module_bucket_plans (module FQNs), nodes_in_subgraph (node lists per plan entry), a ManualOverlapPreservingBucketer, the
    # insert_overlap_deps flag and the module stack function. Collectives get zero size and time estimates. Key methods: run (entry point),
    # _manual_bucket_collectives, _manual_reorder_graph.
    """
    Scheduler that manual buckets and reorders collective nodes based on module_bucket_plans
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        module_bucket_plans: list[list[str] | str],
        insert_overlap_deps: bool,
        module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]] | None = None,
        bucket_mode: BucketMode | None = None,
    ):
        # [FUNCTION DOC] Configures the base scheduler with bucketing on and all memory/overlap limits off, using a no-op runtime estimator
        # (always zero) because estimates are unused. Args: graph module, module plans, whether to insert control dependencies, optional module
        # stack function, optional bucket mode (defaults to custom ops). Creates the bucketer over the whole graph and stores the plan fields.
        # Manual overlap historically used "custom_ops" mode for bucketing
        bucket_mode = bucket_mode or "custom_ops"
        super().__init__(
            gm,
            max_in_flight_gb=0.0,
            max_compute_pre_fetch=0,
            collective_bucketing=True,
            insert_overlap_deps=insert_overlap_deps,
            compute_overlap_multipler=0.0,
            max_coll_distance=0,
            # ManualOverlapScheduler doesn't use runtime estimates (it
            # hardcodes estimated_time_ms=0 in _identify_collectives and
            # schedules purely from module_bucket_plans). Providing a
            # no-op estimator avoids the analytical NCCL path, which
            # crashes in compile-on-one-rank graphs where group_name is
            # an FX Node and the distributed runtime may not be available.
            custom_runtime_estimation=lambda node, size: 0.0,
            collective_estimator="analytical",
            max_memory_increase_gb=None,
            max_memory_increase_ratio=None,
            bucket_mode=bucket_mode,
        )
        self.module_bucket_plans = module_bucket_plans
        self.nodes_in_subgraph: list[list[fx.Node]] = []

        self.bucketer = ManualOverlapPreservingBucketer(
            graph=self.graph,
            collective_info=self.collective_info,
            scheduled=OrderedSet(self.graph.nodes),
            bucket_mode=bucket_mode,
        )
        self.insert_overlap_deps = insert_overlap_deps

        self.module_stack_fn = module_stack_fn

    def _identify_collectives(self) -> None:
        # [FUNCTION DOC] HELPER: Registers every schedulable wait node and its start node as a collective with zero size and zero time estimates,
        # filling collective_info, wait_to_start and the unscheduled collective set.
        """Identify all collective operations."""
        for node in self.nodes:
            if _schedulable_wait_node(node):
                start = node.args[0]
                info = CollectiveInfo(
                    start_node=start,
                    wait_node=node,
                    size_bytes=0,
                    estimated_time_ms=0,
                    exposed_time_ms=0,
                )
                self.collective_info[start] = info
                self.wait_to_start[node] = start
                self.unscheduled_collectives.add(start)

    def _add_to_ready_queue(self, node: fx.Node) -> None:
        # [FUNCTION DOC] HELPER: Pushes the node onto the single ready heap keyed by its original node index, preserving original order.
        """Manual scheduling uses single queue ordered by original node index."""
        heapq.heappush(self.on_path_ready, (self.node_idx[node], node))

    def run(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Entry point: buckets collectives per module plan, then reorders the graph to realize the overlap. Returns the (mutated) graph module.
        """Entry point to run the manual bucket algorithm"""
        # Bucket collectives in each bucket_module
        self._manual_bucket_collectives()

        # Reorder collectives with last/next bucket_module
        self._manual_reorder_graph()

        return self.gm

    def _manual_reorder_graph(self) -> None:
        # [FUNCTION DOC] HELPER: Reorders the graph to enforce manual overlap. Rebuilds the ready queue, then schedules nodes in original order while
        # deferring each bucketed reduce-scatter/all-reduce wait behind the next group's start nodes (recorded as dependencies). Then walks the
        # reversed schedule and makes each bucketed all-gather wait depend on the preceding picked all-gather starts (prefetch); leftover starts
        # attach to the last compute node unless already its ancestor. Finally physically moves the nodes, lints the graph, and, if enabled,
        # inserts control dependencies preserving the order. Side effects: mutates graph and scheduler state.
        """
        Reorder nodes in the FX graph to enforce manual overlap dependencies.

        forward graph (all-gathers only):
            modules are processed in order: module 0, 1, 2, ...

            before reordering:
            ag_start_0 -> ag_wait_0 -> compute_0 -> ag_start_1 -> ag_wait_1 -> compute_1 -> ...

            Reordering prefetches module i+1's parameters while computing module i
            It adds dependencies: ag_wait_i should depend on ag_start_(i+1)
            This enforces ag_start_(i+1) to happen before ag_wait_i so it overlaps with module i's compute

            after reordering:
            ag_start_0 -> ag_start_1 -> ag_wait_0 -> compute_0 -> ag_wait_1 -> compute_1 -> ...

        backward graph (all-gathers and reduce-scatters):
            modules are processed in reverse order: module N, N-1, N-2, ...

            before reordering:
            ag_start_N -> ag_wait_N -> compute_N -> rs_start_N -> rs_wait_N -> ...

            For all-gathers, prefetch module i-1's parameters while computing module i
            Adds dependencies: ag_wait_i should depend on ag_start_(i-1)
            So ag_start_(i-1) overlaps with module i's compute

            For reduce-scatters, defer rs_wait_i to happen after rs_start_(i-1)
            Adds dependencies: rs_wait_i should depend on rs_start_(i-1)
            So rs_start_i overlaps with module i-1's compute

        """
        delayed_rs_wait_nodes: list[fx.Node] = []
        current_rs_start_nodes: list[fx.Node] = []
        overlap_deps: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(OrderedSet)

        # Re-initialize after graph modification in _manual_bucket_collectives
        self.node_idx = {n: i for i, n in enumerate(self.nodes)}
        self.on_path_ready = []
        self.scheduled = OrderedSet()
        for node in self.nodes:
            if self.in_degree[node] == 0:
                self._add_to_ready_queue(node)

        # schedule reduce scatter normally in self._schedule
        while self.on_path_ready:
            _, node = heapq.heappop(self.on_path_ready)
            node_type = self.bucketer.bucketed_node_types.get(node, "")

            if node in self.scheduled:
                continue

            if node_type in ("bucketed_reduce_scatter", "bucketed_all_reduce"):
                # Collect grad-reduction start nodes (pre_bucket_rs and rs, or
                # the HSDP/DDP replicate all_reduce).
                current_rs_start_nodes.append(node)

            elif node_type in (
                "bucketed_reduce_scatter_wait",
                "bucketed_all_reduce_wait",
            ):
                # When we see a wait node from a new RS/AR, flush delayed waits
                # with dependencies on previously collected start nodes
                if current_rs_start_nodes:
                    for delayed in delayed_rs_wait_nodes:
                        for rs_start in current_rs_start_nodes:
                            overlap_deps[delayed].add(rs_start)
                    delayed_rs_wait_nodes.clear()
                    current_rs_start_nodes.clear()
                delayed_rs_wait_nodes.append(node)

            self._schedule(node)

        self.scheduled = OrderedSet(reversed(list(self.scheduled)))
        picked_ag: list[fx.Node] = []
        last_compute: fx.Node | None = None

        for node in self.scheduled:
            node_type = self.bucketer.bucketed_node_types.get(node, "")
            if node_type == "bucketed_all_gather":
                picked_ag.append(node)
                continue

            if node_type == "bucketed_all_gather_wait":
                # Connect corresponding all_gather_wait -> all_gather edges
                if picked_ag:
                    for ag in picked_ag:
                        overlap_deps[self.bucketer.node_to_wait_map[node]].add(ag)
                picked_ag.clear()
            if is_compute_node(node):
                last_compute = node

        if last_compute is not None:
            if not any(
                self.node_ancestors.is_ancestor(ag, last_compute) for ag in picked_ag
            ):
                for ag in picked_ag:
                    overlap_deps[last_compute].add(ag)

        _move_overlap_nodes(self.graph, overlap_deps, self.bucketer.bucketed_node_types)
        self.graph.lint()

        if self.insert_overlap_deps:
            from torch._inductor.fx_passes.control_dependencies import (
                preserve_node_ordering,
            )

            preserve_node_ordering(self.graph, overlap_deps)

    def _manual_bucket_collectives(self) -> None:
        # [FUNCTION DOC] HELPER: Buckets collectives within each module's node list from the plans, lints the graph, then refreshes the node list and
        # in-degree counts from the modified graph. Side effect: mutates the graph.
        # KNOWN ISSUE: the enumerate index in the loop is unused.
        """Bucket nodes in each module_bucket from module_bucket_plans."""
        self._obtain_nodes_in_subgraph()
        for i, nodes in enumerate(self.nodes_in_subgraph):
            self.bucketer.manual_bucket_collectives(nodes=nodes)

        self.graph.lint()
        self.nodes = list(self.graph.nodes)
        self.in_degree = Counter(user for node in self.nodes for user in node.users)

    def _schedule(self, node: fx.Node) -> None:
        # [FUNCTION DOC] HELPER: Marks a node scheduled and queues users whose in-degree drops to zero. Raises an error if already scheduled or if
        # any input is not yet scheduled.
        """Schedule a node."""
        if node in self.scheduled:
            raise AssertionError(f"node already scheduled: {node}")
        if not all(n in self.scheduled for n in node.all_input_nodes):
            raise AssertionError(f"all input nodes must be scheduled before {node}")
        self.scheduled.add(node)
        for user in node.users:
            self.in_degree[user] -= 1
            if self.in_degree[user] == 0:
                self._add_to_ready_queue(user)

    def _obtain_nodes_in_subgraph(self) -> None:
        # [FUNCTION DOC] HELPER: Builds a graph view from the module stack function and fills nodes_in_subgraph with the nodes of each plan entry.
        # Does nothing if no view can be built. Raises an error if any node appears in more than one subgraph.
        """
        Obtain nodes in each subgraph from module_bucket_plans
        """
        graph_view: GraphView | None = make_graph_view(self.graph, self.module_stack_fn)
        if graph_view is None:
            return

        for module in self.module_bucket_plans:
            subgraph_view = get_subgraph_by_path(graph_view, module)
            self.nodes_in_subgraph.append(subgraph_view)

        all_subgraph_nodes = [
            node for sublist in self.nodes_in_subgraph for node in sublist
        ]
        unique_subgraph_nodes = list(OrderedSet(all_subgraph_nodes))
        if len(all_subgraph_nodes) > len(unique_subgraph_nodes):
            raise AssertionError(
                f"Overlapping FX nodes detected across subgraphs in `module_bucket_plans`. "
                f"Expected disjoint node sets but found "
                f"{len(all_subgraph_nodes) - len(unique_subgraph_nodes)} duplicated node(s)."
            )


def manual_overlap_bucketing(
    gm: torch.fx.GraphModule,
    module_bucket_plans: list[list[str] | str],
    insert_overlap_deps: bool = False,
    module_stack_fn: Callable[[fx.Node], list[tuple[str, type[Any]]]] | None = None,
    bucket_mode: BucketMode | None = None,
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Public entry point: runs the manual scheduler on the graph module with the given module bucket plans (bucketing then
    # reordering), recompiles the module and returns it. Args: graph module, module FQN plans, whether to insert control dependencies,
    # optional module stack function and bucket mode. Side effect: mutates and recompiles the input module.
    """Schedule nodes based on user specifications in module_bucket_plans
    The manual overlapping consists of two steps:
    Step 1: bucket all-gather/reduce-scatter in each module in module_bucket_plans
    Step 2: reorder all-gather to overlap with last module_bucket &
        reorder reduce-scatter to overlap with next module_bucket
    TODO(ruisizhang123): allow users to explicitly specify which
        module_bucket they want to overlap.

    Args:
        gm: input graph module to optimize.
        module_bucket_plans: user specified FQNs
        module_stack_fn: Optional callable for extracting module hierarchy from nodes.
            Used to construct a GraphView for identifying nodes in module_bucket_plans.
            The module_class component of the returned tuples is not used by this pass.

            See the `module_stack_fn` parameter in `make_graph_view` (graph_view.py) for
            detailed documentation on signature, return format, and usage examples.
        bucket_mode: Bucket mode for collective bucketing. None uses default.
    """
    # decode abbreviated FQNs to actual FQNs
    overlapped_gm = ManualOverlapScheduler(
        gm,
        module_bucket_plans,
        insert_overlap_deps,
        module_stack_fn,
        bucket_mode=bucket_mode,
    ).run()
    overlapped_gm.recompile()
    return overlapped_gm
