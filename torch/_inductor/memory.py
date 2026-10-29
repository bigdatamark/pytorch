# [MODULE DOC] Memory estimation and memory-aware reordering of Inductor scheduler nodes.
# Defines data classes holding per-buffer / per-node memory planning info, functions that compute buffer sizes
# and lifetimes, estimate peak memory for a node order, and (below) schedule-ordering algorithms (topological,
# min-memory DFS, reorder-for-peak-memory) that try to lower peak memory. Import-time side effect: creates a module logger.
from __future__ import annotations

import collections
import dataclasses
import heapq
import logging
from typing import TYPE_CHECKING, TypedDict

import torch
from torch._environment import is_fbcode
from torch._utils_internal import signpost_event
from torch.utils._ordered_set import OrderedSet

from . import config
from .ir import MultiOutputLayout, NoneLayout
from .utils import get_dtype_size, is_nonfreeable_buffers
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .dependencies import Dep
    from .scheduler import BaseSchedulerNode, SchedulerBuffer

from .dependencies import WeakDep


torch_log = logging.getLogger(__name__)


@dataclasses.dataclass
class PeakMemoryResult:
    # [CLASS DOC] Result of a peak-memory ordering attempt: the node order, its estimated peak memory in bytes, and the name of the method that produced it.
    order: list[BaseSchedulerNode]
    peak_memory: int
    method: str


@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    # [CLASS DOC] Memory planning info for one buffer: bytes allocated at creation (size_alloc), bytes freed at deletion (size_free),
    # successor nodes that determine lifetime (succ_nodes, excludes fake weak deps) and successor nodes used for ordering
    # (succ_nodes_for_ordering, includes them). Invariant: succ_nodes is a subset in size of succ_nodes_for_ordering.
    size_alloc: int = 0
    size_free: int = 0
    # succ_nodes used for buffer lifetime/freeing (excludes is_fake WeakDeps)
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    # succ_nodes used for node ordering (includes is_fake WeakDeps)
    succ_nodes_for_ordering: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Validate (via torch._check) that succ_nodes is not larger than succ_nodes_for_ordering; raises an error otherwise.
        torch._check(
            len(self.succ_nodes) <= len(self.succ_nodes_for_ordering),
            lambda: f"succ_nodes must be a subset of succ_nodes_for_ordering. "
            f"len(succ_nodes)={len(self.succ_nodes)}, len(succ_nodes_for_ordering)={len(self.succ_nodes_for_ordering)}",
        )


@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    # [CLASS DOC] Memory planning info for one scheduler node: its schedule index, bytes it allocates (size), the buffers it reads
    # (pred_buffers), predecessor nodes and successor nodes.
    index: int = 0
    size: int = 0
    pred_buffers: OrderedSet[SchedulerBuffer | FreeableInputBuffer] = dataclasses.field(
        default_factory=OrderedSet
    )
    pred_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )


@dataclasses.dataclass
class FreeableInputBuffer:
    # [CLASS DOC] A graph-input buffer that can be freed after its last use; holds its name and its MemoryPlanningInfoForBuffer (mpi_buffer). Hashable by name.
    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def get_name(self) -> str:
        # [FUNCTION DOC] Return the buffer name.
        return self.name

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash by buffer name.
        return hash(self.name)


def get_freeable_input_buf(
    nodes: list[BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
) -> dict[str, FreeableInputBuffer]:
    # [FUNCTION DOC] Scan nodes' reads for graph inputs that are freeable (not in the non-freeable set) and build a FreeableInputBuffer per
    # name, recording its size hint and successor nodes (for lifetime, skipping fake weak deps; and for ordering, all deps).
    # Returns a mapping from buffer name to FreeableInputBuffer.
    """
    Create and keep track of all input buffers that can be freed during the program

    Returns:
        A dictionary containing all freeable input buffers, keyed by their names.
    """

    def _dep_size_hint(dep: Dep) -> int:
    # [FUNCTION DOC] HELPER: Return the size hint (bytes) of a dependency from the graph.
        return V.graph.get_dep_size_hint(dep)

    # get freeable input buffers' successor nodes for memory lifetime (excludes is_fake WeakDeps)
    # and for ordering (includes all deps)
    dep_name_to_succ_nodes: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_succ_nodes_for_ordering: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_size: dict[str, int] = dict()

    for node in nodes:
        for dep in node.read_writes.reads:
            if dep.name in graph_inputs:
                if not is_nonfreeable_buffers(dep):
                    # All deps contribute to ordering, but fake weak deps do not contribute to
                    # memory liveness
                    dep_name_to_succ_nodes_for_ordering[dep.name].add(node)
                    dep_name_to_size[dep.name] = _dep_size_hint(dep)
                    if not (isinstance(dep, WeakDep) and dep.is_fake):
                        dep_name_to_succ_nodes[dep.name].add(node)

    # create FreeableInputBuffer objects and add them to the returned dictionary
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer] = dict()
    for dep_name in dep_name_to_succ_nodes_for_ordering:
        name_to_freeable_input_buf[dep_name] = FreeableInputBuffer(
            dep_name,
            MemoryPlanningInfoForBuffer(
                size_free=dep_name_to_size[dep_name],
                succ_nodes=dep_name_to_succ_nodes[dep_name],
                succ_nodes_for_ordering=dep_name_to_succ_nodes_for_ordering[dep_name],
            ),
        )
    return name_to_freeable_input_buf


def compute_size_for_scheduler_buffer(
    name_to_buf: dict[str, SchedulerBuffer],
) -> dict[str, tuple[int, int]]:
    # [FUNCTION DOC] Compute (size_alloc, size_free) in bytes for every scheduler buffer. Mutated-name and NoneLayout buffers cost zero;
    # a MultiOutputLayout buffer allocates the sum of its MultiOutput users' sizes while those users allocate zero and free their own
    # size; other buffers allocate and free numel times dtype size. Returns a mapping from buffer name to the tuple.
    """
    Compute the size of each scheduler buffer, including (1) memory allocated when
    it is created and (2) memory deallocated when it is freed.

    We specially handle the case of MultiOutputLayout.
    Consider the following case:
        buf0 = some_ops_with_multi_outputs(...)
        buf1 = buf0[0] # assume 10 bytes
        buf2 = buf0[1] # assume 20 bytes
    In such cases,
        buf0: at creation, 30 bytes allocated, when deleted, 0 bytes freed
        buf1: at creation, 0 bytes allocated, when deleted, 10 bytes freed
        buf2: at creation, 0 bytes allocated, when deleted, 20 bytes freed

    When an operation mutates a buffer in-place, the scheduler creates a new buffer name
    to track the "before" and "after" states, even though they share the same memory.

    The mutated buffer represents a rename with zero allocation and deallocation cost.
    During dependency tracking, we transfer dependencies from the mutated name back to
    the original buffer, ensuring the original memory is only freed when all aliases
    are done.

    This handles cases where a buffer has multiple non-overlapping aliases - rather than
    trying to assign free costs to individual aliases, we forward all alias dependencies
    to the original buffer.

    Consider:
        buf0 = op0()
        buf1 = mutation_op_(buf0)
        del buf0
        ...
        op(buf1)
        del buf1

    The only memory events are the creation prior to op0, and the deletion following buf1.

    Returns:
        A dictionary mapping a scheduler buffer to a tuple of (size_alloc, size_free).
    """
    from .ir import MultiOutput
    from .scheduler import OutputNode

    sched_buf_to_size: dict[str, tuple[int, int]] = dict()

    def _compute_and_update_buf_size(
        sched_buf: SchedulerBuffer, user_of_MultiOutputLayout: bool = False
    ) -> int:
        # [FUNCTION DOC] HELPER: Recursively compute and record one buffer's (alloc, free) sizes in the enclosing mapping and return
        # the size it accounts for. user_of_MultiOutputLayout makes the buffer allocate zero since its parent allocated it.
        if sched_buf.get_name() in V.graph.scheduler.mutation_real_name:
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, NoneLayout):
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, MultiOutputLayout):
            size_alloc = 0
            for user in sched_buf.users:
                if isinstance(user.node, OutputNode):
                    continue
                for buf in user.node.get_outputs():
                    if isinstance(buf.node, MultiOutput):
                        size_alloc += _compute_and_update_buf_size(buf, True)
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else size_alloc,
                0,
            )
            return size_alloc
        else:
            buf_size = V.graph.sizevars.optimization_hint(
                sched_buf.node.get_numel(), fallback=0
            ) * get_dtype_size(sched_buf.node.get_dtype())
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else buf_size,
                buf_size,
            )
            return buf_size

    for sched_buf in name_to_buf.values():
        # skip if sched_buf is already processed as a user of another SchedulerBuffer
        # whose layout is of the type MultiOutputLayout
        if sched_buf.get_name() not in sched_buf_to_size:
            _compute_and_update_buf_size(sched_buf)

    return sched_buf_to_size


def assign_memory_planning_info_for_scheduler_buffers(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
) -> None:
    # [FUNCTION DOC] Fill the mpi_buffer attribute of each scheduler buffer: sizes plus successor nodes derived from nodes' unmet dependencies.
    # Successors of in-place-mutated names are merged into the original real buffer (processed in reverse so merges are transitive).
    # Mutates the buffers; returns None.
    """
    For each SchedulerBuffer, assign its size info and successor nodes.
    A buffer's successor nodes determines when a buffer can be freed.
    """
    # get buffer sizes
    sched_buf_to_size = compute_size_for_scheduler_buffer(name_to_buf)

    # get buffer's successor nodes for memory lifetime (excludes is_fake WeakDeps)
    # and for ordering (includes all deps)
    dep_name_to_succ_nodes: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_succ_nodes_for_ordering: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    for node in nodes:
        for dep in node.unmet_dependencies:
            # All deps contribute to ordering, but fake weak deps do not contribute to
            # memory liveness
            dep_name_to_succ_nodes_for_ordering[dep.name].add(node)
            if not (isinstance(dep, WeakDep) and dep.is_fake):
                dep_name_to_succ_nodes[dep.name].add(node)

    # iterate in reverse, so dependencies are picked up transitively.
    for mutating_buf_name, real_buf_name in reversed(
        V.graph.scheduler.mutation_real_name.items()
    ):
        dep_name_to_succ_nodes[real_buf_name] |= dep_name_to_succ_nodes[
            mutating_buf_name
        ]
        dep_name_to_succ_nodes_for_ordering[real_buf_name] |= (
            dep_name_to_succ_nodes_for_ordering[mutating_buf_name]
        )

    # populate the MemoryPlanningInfoForBuffer attribute to each scheduler buffer
    # note: there are scheduler buffers not in dep_name_to_succ_nodes (e.g., graph outputs)
    for buf_name in name_to_buf:
        name_to_buf[buf_name].mpi_buffer = MemoryPlanningInfoForBuffer(
            size_alloc=sched_buf_to_size[buf_name][0],
            size_free=sched_buf_to_size[buf_name][1],
            succ_nodes=dep_name_to_succ_nodes[buf_name],
            succ_nodes_for_ordering=dep_name_to_succ_nodes_for_ordering[buf_name],
        )


def assign_memory_planning_info_for_scheduler_nodes(
    nodes: list[BaseSchedulerNode],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None:
    # [FUNCTION DOC] Fill the mpi_node attribute of each node: its index, allocated size, predecessor/successor nodes (derived from output buffers'
    # successor sets, with self links removed) and predecessor buffers (including freeable input buffers). Mutates nodes; returns None.
    """
    Assign to each scheduler node its predecessor and successor nodes.
    """

    node_to_pred_nodes: dict[BaseSchedulerNode, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    node_to_succ_nodes: dict[BaseSchedulerNode, OrderedSet[BaseSchedulerNode]] = {}
    node_to_pred_buffers: dict[
        BaseSchedulerNode, OrderedSet[SchedulerBuffer | FreeableInputBuffer]
    ] = collections.defaultdict(OrderedSet)

    # collect all predecessors using existing successor mappings
    for node in nodes:
        succ_nodes = OrderedSet(
            succ_node
            for buffer in node.get_outputs()
            for succ_node in buffer.mpi_buffer.succ_nodes_for_ordering
        )
        node_to_succ_nodes[node] = succ_nodes

        # For each successor, add current node as its predecessor
        for succ_node in succ_nodes:
            node_to_pred_nodes[succ_node].add(node)

        # For each output buffer, add it as predecessor to its successor nodes
        # Use succ_nodes (not succ_nodes_for_ordering) since pred_buffers is used
        # for memory lifetime tracking, not ordering
        for buffer in node.get_outputs():
            for succ_node in buffer.mpi_buffer.succ_nodes:
                node_to_pred_buffers[succ_node].add(buffer)

    for freeable_buffer in name_to_freeable_input_buf.values():
        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:
            node_to_pred_buffers[succ_node].add(freeable_buffer)

    # Second pass: assign memory planning info using completed predecessor mappings
    for index, node in enumerate(nodes):
        size_alloc = sum(buffer.mpi_buffer.size_alloc for buffer in node.get_outputs())
        succ_nodes = node_to_succ_nodes[node]
        pred_nodes = node_to_pred_nodes[node]

        # make sure we do not make node a successor or predecessor of itself
        succ_nodes.discard(node)
        pred_nodes.discard(node)

        node.mpi_node = MemoryPlanningInfoForNode(
            index=index,
            size=size_alloc,
            pred_buffers=node_to_pred_buffers[node],
            pred_nodes=node_to_pred_nodes[node],
            succ_nodes=succ_nodes,
        )


# map each scheduler buffer to its size, start step, and end step
@dataclasses.dataclass
class BufferInfo:
    # [CLASS DOC] Lifetime record of one buffer: alloc and free sizes and start step and end step in the schedule (end_step is -1 for graph outputs, which are never freed).
    buffer: SchedulerBuffer | FreeableInputBuffer
    size_alloc: int
    size_free: int
    start_step: int
    end_step: int


def compute_memory_timeline(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    list[BufferInfo],
    dict[BaseSchedulerNode, int],
    dict[FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode],
]:
    # [FUNCTION DOC] Compute a BufferInfo for every freeable input buffer (live from step 0) and every scheduler buffer (live from its defining step) with
    # end step as the last successor step (the defining step if unused, -1 for graph outputs). Returns the list of BufferInfo, a mapping from node to step, and a
    # mapping from buffer to the node of its last use. Raises AssertionError if expected last-use nodes are missing.
    """
    Compute buffer allocation and deallocation sizes and map their
    lifetime to the node schedule
    """

    # get the execution step of each node, this will be used to determine
    # the end_step of buffers
    node_to_step: dict[BaseSchedulerNode, int] = {
        node: step for step, node in enumerate(nodes)
    }

    # get buffers' size and liveliness information
    buf_info_list: list[BufferInfo] = []
    buf_to_snode_last_use: dict[
        FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode
    ] = {}

    def _get_end_step_and_snode(
        buf: FreeableInputBuffer | SchedulerBuffer,
    ) -> tuple[int, BaseSchedulerNode | None]:
        # [FUNCTION DOC] HELPER: Return the largest step among a buffer's successor nodes and that node, or (-1, None) if it has no successors.
        max_step: int = -1
        max_step_snode: BaseSchedulerNode | None = None
        succ_nodes = buf.mpi_buffer.succ_nodes
        if succ_nodes:
            for succ_node in succ_nodes:
                step = node_to_step[succ_node]
                if step > max_step:
                    max_step = step
                    max_step_snode = succ_node
            if max_step_snode is None:
                raise AssertionError("expected max_step_snode to be set")
        return max_step, max_step_snode

    # 1. for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        end_step = -1
        if buf_name not in graph_outputs:
            end_step, end_step_snode = _get_end_step_and_snode(input_buf)
            if end_step_snode is None:
                raise AssertionError("expected end_step_snode to be set")
            buf_to_snode_last_use[input_buf] = end_step_snode

        buf_info_list.append(
            BufferInfo(
                input_buf,
                input_buf.mpi_buffer.size_free,
                input_buf.mpi_buffer.size_free,
                0,
                end_step,
            )
        )

    # 2. for scheduler buffers
    for step, node in enumerate(nodes):
        for sched_buf in node.get_outputs():
            # note: it is possible for a non-graph-output sched_buf to have no succ_nodes and
            # to be only used by its defining op (e.g., due to fusion when all consumers of
            # the buffer are fused with its defining op). In such cases, end_step is step.
            buf_name = sched_buf.get_name()
            end_step = -1
            if buf_name not in graph_outputs:
                end_step, end_step_snode = _get_end_step_and_snode(sched_buf)
                if end_step == -1:
                    end_step = step
                    buf_to_snode_last_use[sched_buf] = node
                else:
                    if end_step_snode is None:
                        raise AssertionError("expected end_step_snode to be set")
                    buf_to_snode_last_use[sched_buf] = end_step_snode

            buf_info_list.append(
                BufferInfo(
                    sched_buf,
                    sched_buf.mpi_buffer.size_alloc,
                    sched_buf.mpi_buffer.size_free,
                    step,
                    end_step,
                )
            )

    return buf_info_list, node_to_step, buf_to_snode_last_use


def peak_memory_from_buf_info_list(
    buf_info_list: list[BufferInfo], num_steps: int
) -> tuple[int, list[int]]:
    # [FUNCTION DOC] From buffer lifetimes build per-step allocation deltas (frees are skipped for end_step -1), then accumulate. Returns the peak
    # memory and the list of live memory after each step (length num_steps plus one).
    """Compute peak and per-step live memory from buffer lifetimes.

    Skip the free for graph-output buffers (`end_step == -1`); otherwise
    `delta[0] -= size_free` would cancel the alloc.
    """
    delta = [0] * (num_steps + 1)
    for bi in buf_info_list:
        delta[bi.start_step] += bi.size_alloc
        if bi.end_step != -1:
            delta[bi.end_step + 1] -= bi.size_free

    max_memory = 0
    cur_memory = 0
    memories_at_nodes = [0] * (num_steps + 1)
    for t in range(num_steps + 1):
        cur_memory += delta[t]
        memories_at_nodes[t] = cur_memory
        if cur_memory > max_memory:
            max_memory = cur_memory
    return max_memory, memories_at_nodes


def live_memory_before_steps_from_buf_info_list(
    buf_info_list: list[BufferInfo], num_steps: int
) -> list[int]:
    # [FUNCTION DOC] Return live bytes before each step runs: freeable inputs count from the start, other buffers become live the step after their start step,
    # and frees apply after the end step. Result has num_steps plus one entries.
    """Compute live bytes before each scheduler step runs."""
    delta = [0] * (num_steps + 1)
    cur_memory = 0
    for bi in buf_info_list:
        if isinstance(bi.buffer, FreeableInputBuffer):
            cur_memory += bi.size_alloc
        else:
            delta[bi.start_step + 1] += bi.size_alloc

        if bi.end_step != -1:
            delta[bi.end_step + 1] -= bi.size_free

    live_before = [0] * (num_steps + 1)
    live_before[0] = cur_memory
    for t in range(1, num_steps + 1):
        cur_memory += delta[t]
        live_before[t] = cur_memory
    return live_before


def estimate_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[int, list[int]]:
    # [FUNCTION DOC] Estimate peak memory of nodes in the given execution order using buffer lifetimes. Returns (peak memory, memory per step).
    """
    Given a list of nodes in their execution order, estimate the peak memory, by
    keeping track of the liveliness of SchedulerBuffers and FreeableInputBuffers.

    Returns:
        int: peak memory
        List[int]: memory usage at each node (or each step).
    """
    buf_info_list, _, _ = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )
    return peak_memory_from_buf_info_list(buf_info_list, len(nodes))


def estimate_region_peak_memory(
    nodes_in_window: Iterable[BaseSchedulerNode],
    *,
    region_start: int,
    region_end: int,
    step_of: Callable[[BaseSchedulerNode], int],
    graph_outputs: OrderedSet[str],
    cur_memory: int = 0,
) -> int:
    # [FUNCTION DOC] Estimate the peak memory within a window of the schedule for a hypothetical order. For each node (step from step_of) add its outputs' alloc
    # and free outputs with no successors, and free pred buffers whose last consumer is this node (graph outputs are never freed); then scan from cur_memory (live bytes at the
    # window start), allocating before freeing. Raises AssertionError if a node's step is outside the region or a pred buffer has no successors. Returns the peak.
    """Peak memory inside `[region_start, region_end]` for the
    hypothetical post-reorder schedule.

    Walks `nodes_in_window` (post-rewrite, in proposed order). For
    each node: alloc = sum of `size_alloc` over its outputs; free =
    sum of `size_free` over `pred_buffers` whose proposed last
    consumer is this node. Then accumulates per step starting from
    `cur_memory` (live bytes at the window boundary) and returns
    the maximum live bytes.
    """
    R = region_end - region_start + 1
    region = [SNodeMemory(0, 0) for _ in range(R)]

    for node in nodes_in_window:
        s = step_of(node)
        slot = s - region_start
        if not (0 <= slot < R):
            raise AssertionError(f"expected 0 <= slot < {R}, got {slot}")

        for buf in node.get_outputs():
            bi = buf.mpi_buffer
            region[slot].size_alloc += bi.size_alloc
            name = buf.get_name()
            if name in graph_outputs:
                continue
            succ_steps = [step_of(n) for n in bi.succ_nodes]
            if not succ_steps:
                region[slot].size_free += bi.size_free

        for pb in node.mpi_node.pred_buffers:
            name = pb.get_name()
            if name in graph_outputs:
                continue
            succ_steps = [step_of(n) for n in pb.mpi_buffer.succ_nodes]
            if not succ_steps:
                raise AssertionError("expected non-empty succ_steps")
            if max(succ_steps) == s:
                region[slot].size_free += pb.mpi_buffer.size_free

    cur = cur_memory
    peak = cur
    for af in region:
        cur += af.size_alloc
        if cur > peak:
            peak = cur
        cur -= af.size_free
    return peak


@dataclasses.dataclass
class SNodeMemory:
    # [CLASS DOC] Bytes allocated and bytes freed at one scheduler step.
    size_alloc: int
    size_free: int


def estimate_peak_memory_allocfree(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    int,
    list[tuple[int, int]],
    dict[BaseSchedulerNode, SNodeMemory],
    dict[FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode],
]:
    # [FUNCTION DOC] Variant of estimate_peak_memory that tracks alloc and free separately per node (peak is after alloc, before free).
    # Returns peak memory, per-step (post-alloc, post-free) memory, mapping from node to SNodeMemory, and mapping from buffer to the node of its last use.
    """
    Alternative version of estimate_peak_memory, that respects the fact,
    that every SchedulerNode has multiple phases:
    1. alloc ( outputs )
    2. run_kernel
    3. dealloc last_use buffers
    estimate_peak_memory collapses memory into one value: size_alloc - size_free
    While peak memory happens after alloc.

    Duplicating the code to not migrate all callsites at once,
    In future usages of estimate_peak_memory will migrate to this version.
    """

    buf_info_list, _, buf_to_snode_last_use = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    # incremental memory changes at each step
    step_idx_allocfree = [SNodeMemory(0, 0) for _ in range(len(nodes))]

    # for each buffer, update memory when created and when freed
    for buf_info in buf_info_list:
        step_idx_allocfree[buf_info.start_step].size_alloc += buf_info.size_alloc
        if buf_info.end_step != -1:
            step_idx_allocfree[buf_info.end_step].size_free += buf_info.size_free

    snodes_allocfree = {}
    for i, node in enumerate(nodes):
        snodes_allocfree[node] = step_idx_allocfree[i]

    max_memory = 0
    cur_memory = 0
    snodes_curr_memory = []
    for t in range(len(nodes)):
        alloc = step_idx_allocfree[t].size_alloc
        free = step_idx_allocfree[t].size_free
        cur_memory += alloc
        post_alloc = cur_memory
        max_memory = max(max_memory, cur_memory)
        cur_memory -= free
        post_free = cur_memory
        snodes_curr_memory.append((post_alloc, post_free))

    return (
        max_memory,
        snodes_curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
    )


def topological_sort_lpmf(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_buf: dict[str, SchedulerBuffer],
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    # [FUNCTION DOC] Greedy "least peak memory first" topological sort. Tracks per-node unmet predecessors and per-buffer remaining users, keeps
    # live and max memory, and at each step picks among ready nodes: if config.size_threshold_for_succ_based_strategy is positive and all ready nodes
    # are larger than it, the node whose earliest successor comes first; otherwise the node minimizing (size if above the memory gap else 0, net
    # memory growth, original index). Updates memory-to-free as buffers approach their last use. Raises AssertionError on inconsistent degrees.
    # Returns the new node order. KNOWN ISSUE: the final check that iterations exceed the node count can never be true because the loop is bounded.
    """
    A bfs-based greedy topological order. LPMF stands for "Least Peak Memory First".

    The idea is from this paper:
    Buffer memory optimization for video codec application modeled in Simulink
    https://www.cs.york.ac.uk/rts/docs/DAC-1964-2006/PAPERS/2006/DAC06/PDFFILES/P0689.PDF

    The algorithm maintains the max memory so far.
    At every iteration, for each scheduleable node, it computes:
        - how much memory needs to be allocated for the output buffers of this node;
        - how much memory can be freed as a result of executing this node.
    This gives us two values for each node:
        (1) mem1: memory during the execution of the node;
        (2) mem2: memory after executing the node, after some input buffers are freed.
    The greedy approach select as follows:
        (i) if there are nodes whose mem1 values are below the max memory so far,
            then pick the node with the lowest mem2 value;
        (ii) otherwise, pick the one with the lowest mem1 value.
    """

    class NodeInfo(TypedDict):
        # [CLASS DOC] HELPER: TypedDict of scheduling state per node: remaining unmet predecessor count (indegree) and bytes freed when the node runs.
        indegree: int
        memory_to_free: int

    class BufferInfo(TypedDict):
        # [CLASS DOC] HELPER: TypedDict of per-buffer state: number of remaining users (outdegree). Shadows the module-level BufferInfo dataclass inside this function.
        outdegree: int

    node_info: dict[BaseSchedulerNode, NodeInfo] = dict()
    buf_info: dict[SchedulerBuffer | FreeableInputBuffer, BufferInfo] = dict()

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: OrderedSet[BaseSchedulerNode] = OrderedSet()
    for node in nodes:
        node_info[node] = {
            "indegree": len(node.mpi_node.pred_nodes),
            "memory_to_free": 0,
        }
        if node_info[node]["indegree"] == 0:
            nodes_to_schedule.add(node)

    # compute buffers' number of unmet successors (used to decide when to free)
    for buf in list(name_to_buf.values()) + list(name_to_freeable_input_buf.values()):
        buf_info[buf] = {
            "outdegree": len(buf.mpi_buffer.succ_nodes)
            + (1 if buf.get_name() in graph_outputs else 0)
        }

    # initialize memory estimations
    live_memory = sum(
        input_buf.mpi_buffer.size_free
        for input_buf in name_to_freeable_input_buf.values()
    )

    # this is the total output memory, which is a lower bound for peak memory
    # we do not include the memory of non freeable input buffers
    output_memory = 0
    for buf_name in graph_outputs:
        if buf_name in name_to_buf:
            output_memory += name_to_buf[buf_name].mpi_buffer.size_free
        elif buf_name in name_to_freeable_input_buf:
            output_memory += name_to_freeable_input_buf[buf_name].mpi_buffer.size_free
    max_memory = max(live_memory, output_memory)
    memory_gap = max_memory - live_memory

    # compute the amount of memory that is allocated when a node is scheduled
    # and the amount of memory that can be freed when a node is scheduled
    for node in nodes:
        # 1. if a buffer read by this node is last used by this node
        for buf in node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] == 1:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free
        # 2. if a buffer written by this node is used internally and not used later
        for buf in node.get_outputs():
            if buf_info[buf]["outdegree"] == 0:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free

    # schedule nodes one at a time
    schedule: list[BaseSchedulerNode] = []
    size_threshold = config.size_threshold_for_succ_based_strategy
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule:
        if (
            size_threshold > 0
            and min(node.mpi_node.size for node in nodes_to_schedule) > size_threshold
        ):
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: min(
                    (
                        succ_node.mpi_node.index
                        for succ_node in node.mpi_node.succ_nodes
                    ),
                    default=len(nodes),
                ),
            )
        else:
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: (
                    node.mpi_node.size if node.mpi_node.size > memory_gap else 0,
                    node.mpi_node.size - node_info[node]["memory_to_free"],
                    node.mpi_node.index,
                ),
            )
        nodes_to_schedule.remove(selected_node)
        schedule.append(selected_node)
        num_iters += 1

        # update memory usage
        live_memory += selected_node.mpi_node.size
        max_memory = max(max_memory, live_memory)
        live_memory -= node_info[selected_node]["memory_to_free"]
        memory_gap = max_memory - live_memory

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi_node.succ_nodes:
            if node_info[succ_node]["indegree"] <= 0:
                raise AssertionError(
                    f"expected positive indegree, got {node_info[succ_node]['indegree']}"
                )
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                nodes_to_schedule.add(succ_node)

        # update predecessor nodes
        for buf in selected_node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] <= 0:
                raise AssertionError(
                    f"expected positive outdegree, got {buf_info[buf]['outdegree']}"
                )
            buf_info[buf]["outdegree"] -= 1
            if buf_info[buf]["outdegree"] == 1:
                for succ_node in buf.mpi_buffer.succ_nodes:
                    node_info[succ_node]["memory_to_free"] += buf.mpi_buffer.size_free

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for lpmf")

    return schedule


def topological_sort_bfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    # [FUNCTION DOC] FIFO-style BFS topological sort: among ready nodes, a heap picks the one whose predecessors executed earliest (sorted list of
    # predecessor execution orders, ties by original index). Returns the new order. Raises AssertionError on inconsistent indegree and RuntimeError if the loop overruns.
    """
    A BFS topological sort that selects nodes whose dependencies are executed the
    earliest. This follows a FIFO idea. Specifically, at every iteration, for each node
    that is schedulable, we gather the order in which its predecessor nodes are executed,
    and this sorted list of execution orders of predecessor nodes defines the priority.
    We select the node whose predecessors nodes are executed the earliest. The FIFO
    idea aims to reduce the liveness duration of buffers created.
    """

    class NodeInfo(TypedDict):
        # [CLASS DOC] HELPER: TypedDict of per-node state: remaining unmet predecessors (indegree) and its position in the new schedule (order, -1 until scheduled).
        indegree: int
        order: int

    node_info: dict[BaseSchedulerNode, NodeInfo] = dict()

    @dataclasses.dataclass
    class NodeWithPriority:
        # [CLASS DOC] HELPER: Heap entry pairing a node with its priority list; ordering compares priority lists, breaking ties by the node's original schedule index.
        priority: list[int]
        node: BaseSchedulerNode

        def __lt__(self, other: NodeWithPriority) -> bool:
            # [FUNCTION DOC] HELPER: Less-than for heap ordering: compare priority lists, falling back to original index when equal.
            if self.priority == other.priority:
                return self.node.mpi_node.index < other.node.mpi_node.index
            return self.priority < other.priority

    def _node_priority(node: BaseSchedulerNode) -> list[int]:
        # [FUNCTION DOC] HELPER: Return the sorted unique execution orders of a ready node's predecessors. Raises AssertionError if the node still has unmet predecessors.
        # priority is the order in which predecessor nodes are executed
        if node_info[node]["indegree"] != 0:
            raise AssertionError(
                f"expected zero indegree, got {node_info[node]['indegree']}"
            )
        exec_orders = sorted(
            OrderedSet(
                node_info[pred_node]["order"] for pred_node in node.mpi_node.pred_nodes
            )
        )
        return exec_orders

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: list[NodeWithPriority] = []
    for node in nodes:
        node_info[node] = {"indegree": len(node.mpi_node.pred_nodes), "order": -1}
        if node_info[node]["indegree"] == 0:
            heapq.heappush(
                nodes_to_schedule, NodeWithPriority(_node_priority(node), node)
            )

    # schedule nodes one at a time
    schedule: list[BaseSchedulerNode] = []
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule
        selected_node = heapq.heappop(nodes_to_schedule).node
        node_info[selected_node]["order"] = len(schedule)
        schedule.append(selected_node)
        num_iters += 1

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi_node.succ_nodes:
            if node_info[succ_node]["indegree"] <= 0:
                raise AssertionError(
                    f"expected positive indegree, got {node_info[succ_node]['indegree']}"
                )
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                heapq.heappush(
                    nodes_to_schedule,
                    NodeWithPriority(_node_priority(succ_node), succ_node),
                )

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for bfs")

    return schedule


def topological_sort_dfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    # [FUNCTION DOC] DFS topological sort that visits nodes in ascending priority (own size plus sizes of buffers read) and recursively emits each node's
    # dependencies (also ordered by that priority) before the node. Returns the new order.
    """
    This is a DFS topological sort. The setup is similar to `topological_sort_schedule`
    in scheduler.py. The difference is the order nodes are visited in the outer loop.
    In `topological_sort_schedule`, nodes are visited in their original order.
    In this function, nodes are visited based on their priority -- for each node, we
    compute the total memory of all buffers it reads from or writes to, and we visit
    the nodes in ascending order of this priority.
    """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: dict[str, BaseSchedulerNode] = dict()
    result: list[BaseSchedulerNode] = []
    size_with_reads: dict[BaseSchedulerNode, int] = dict()

    def visit(n: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] HELPER: Depth-first visit; marks n seen, visits its unmet-dependency producer nodes in priority order, then appends n to the result.
        if n not in seen:
            seen.add(n)
            dep_nodes = [
                name_to_node[dep.name]
                for dep in n.unmet_dependencies
                if dep.name in name_to_node
            ]
            for node in sorted(
                dep_nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)
            ):
                visit(node)
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node

    for node in nodes:
        size_with_reads[node] = node.mpi_node.size + sum(
            pred_buf.mpi_buffer.size_free for pred_buf in node.mpi_node.pred_buffers
        )
    for node in sorted(nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)):
        visit(node)

    return result


def validate_graph_acyclic(nodes: list[BaseSchedulerNode]) -> None:
    # [FUNCTION DOC] Check the predecessor graph of nodes for cycles with a white/gray/black depth-first search. Raises RuntimeError with the cycle path
    # when found, and AssertionError if a node is its own predecessor. Returns None.
    """
    Validate that the graph is acyclic by checking predecessor relationships.

    Raises:
        RuntimeError: If a cycle is detected in the graph
    """
    # DFS coloring scheme for cycle detection:
    # WHITE (0): Node has not been visited yet
    # GRAY (1): Node is currently being processed (in the recursion stack)
    # BLACK (2): Node has been completely processed (finished exploring all its predecessors)
    # A back edge (cycle) is detected when we encounter a GRAY node during DFS traversal
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)
    path: list[BaseSchedulerNode] = []  # Track current DFS path

    def dfs_visit(node: BaseSchedulerNode) -> None:
    # [FUNCTION DOC] HELPER: Recursive coloring DFS step over predecessor nodes; raises RuntimeError when it reaches a node currently on the path.
        if color[node] == BLACK:
            return

        if color[node] == GRAY:
            path.append(node)
            path_info = " -> ".join([node.get_name() for node in path])

            raise RuntimeError(
                f"Cycle detected in memory planning graph"
                f"Path containing cycle (i -> j: j is a dependency of i): {path_info} "
                f"This indicates invalid dependency relationships in the scheduler graph"
            )

        color[node] = GRAY
        path.append(node)

        for pred_node in node.mpi_node.pred_nodes:
            if pred_node == node:
                raise AssertionError("expected pred_node != node (self-loop)")
            dfs_visit(pred_node)

        path.pop()
        color[node] = BLACK

    # Start DFS from all unvisited nodes
    for node in nodes:
        if color[node] == WHITE:
            dfs_visit(node)


def validate_unique_buffer_names(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None:
    # [FUNCTION DOC] Verify every node output buffer is present in name_to_buf and maps to the same buffer object, and that no output name collides with a freeable
    # input buffer name. Raises RuntimeError describing the problem. Returns None.
    """
    Validate that for each node's output buffer, the name_to_buf mapping is correct.
    For each output buffer buf, we should have name_to_buf[buf.get_name()] == buf.
    Also validate that no buffer names overlap with freeable input buffer names.

    Raises:
        RuntimeError: If buffer name mapping is incorrect or names overlap
    """
    for node in nodes:
        for buf in node.get_outputs():
            buf_name = buf.get_name()

            # Check if buffer name exists in the mapping
            if buf_name not in name_to_buf:
                raise RuntimeError(
                    f"{buf_name} from {node.get_name()} is not found in name_to_buf mapping."
                    f" This indicates a missing buffer mapping."
                )

            # Check if the mapping points to the correct buffer object
            if name_to_buf[buf_name] != buf:
                raise RuntimeError(
                    f"Buffer name mapping is incorrect for '{buf_name}'."
                    f"Expected name_to_buf['{buf_name}'] to be {buf.debug_str()}"
                    f"but got {name_to_buf[buf_name].debug_str()}"
                    f"This indicates some buffers share the same name"
                )

            # Check if buffer name conflicts with freeable input buffer names
            if buf_name in name_to_freeable_input_buf:
                raise RuntimeError(
                    f"Buffer name conflict detected: '{buf_name}' from node {node.get_name()} "
                    f"is also used as a freeable input buffer name. "
                )


def prepare_planning_info(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> tuple[int, dict[str, FreeableInputBuffer]]:
    # [FUNCTION DOC] Compute freeable input buffers, assign memory planning info to buffers and nodes, and estimate peak memory for the current order.
    # Returns (estimated peak memory, mapping from name to FreeableInputBuffer). Mutates mpi_buffer / mpi_node attributes.
    """
    Prepare planning info. As nodes are scheduled one at a time, these help
    keep track of when a buffer can be freed, and when a node can be scheduled

    Returns:
        int: peak memory estimation
        dict[str, FreeableInputBuffer]: name to freeable input buffer
    """
    name_to_freeable_input_buf = get_freeable_input_buf(nodes, graph_inputs)
    assign_memory_planning_info_for_scheduler_buffers(nodes, name_to_buf)
    assign_memory_planning_info_for_scheduler_nodes(
        nodes, name_to_fused_node, name_to_buf, name_to_freeable_input_buf
    )

    # the default
    estimated_peak_memory, _ = estimate_peak_memory(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    return estimated_peak_memory, name_to_freeable_input_buf


def reorder_for_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
    methods: list[Callable[..., list[BaseSchedulerNode]]] = [  # noqa: B006
        topological_sort_lpmf,
        topological_sort_bfs,
        topological_sort_dfs,
    ],
) -> list[BaseSchedulerNode]:
    # [FUNCTION DOC] Reorder nodes to reduce estimated peak memory. Prepares planning info, optionally exports the graph for a simulator (config.reorder_for_peak_memory_debug),
    # validates the graph (errors are logged and re-raised except in fbcode), then tries each method (lpmf takes extra args, others only nodes), checks the order length,
    # estimates peak memory, logs results and emits a signpost event. Returns the order with the lowest estimate, the baseline included.
    # Failures of an individual method are logged and re-raised outside fbcode. KNOWN ISSUE: mutable default list for methods (lint-suppressed, never mutated).
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    """

    torch_log.info("Reordering for peak memory -- %d nodes", len(nodes))

    estimated_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
        nodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )

    # export graph for simulator if needed
    if config.reorder_for_peak_memory_debug:
        export_graph_for_simulator(
            nodes,
            name_to_freeable_input_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
        )

    # Validate planning info before proceeding with reordering
    try:
        validate_graph_acyclic(nodes)
        validate_unique_buffer_names(nodes, name_to_buf, name_to_freeable_input_buf)
    except RuntimeError:
        torch_log.exception("Memory planning validation failed")
        if not is_fbcode():  # TODO: remove after ensuring OSS side is safe
            raise

    # keep track of the peak memory estimates of different methods
    peak_memory_diff_methods: list[PeakMemoryResult] = []
    peak_memory_diff_methods.append(
        PeakMemoryResult(nodes, estimated_peak_memory, "baseline")
    )
    torch_log.info("Baseline peak memory: %d", estimated_peak_memory)

    # other methods
    for method in methods:
        try:
            if method is topological_sort_lpmf:
                order = method(
                    nodes, name_to_freeable_input_buf, name_to_buf, graph_outputs
                )
            else:
                order = method(nodes)
            if len(order) != len(nodes):
                raise AssertionError(
                    f"expected order length {len(nodes)}, got {len(order)}"
                )
            peak_memory, _ = estimate_peak_memory(
                order, name_to_freeable_input_buf, graph_outputs
            )
            peak_memory_diff_methods.append(
                PeakMemoryResult(order, peak_memory, method.__name__)
            )
            torch_log.info("%s peak memory: %d", method.__name__, peak_memory)
        except Exception:
            torch_log.exception("Failed to reorder for %s", method.__name__)
            if not is_fbcode():  # TODO: remove after ensuring OSS side is safe
                raise

    signpost_event(
        category="inductor",
        name="memory",
        parameters={
            "orm": {elem.method: elem.peak_memory for elem in peak_memory_diff_methods},
        },
    )

    # get the optimal one
    best_result = min(peak_memory_diff_methods, key=lambda x: x.peak_memory)

    return best_result.order


def export_graph_for_simulator(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> None:
    # [FUNCTION DOC] Debugging aid: build a JSON description of the graph (buffers with sizes, input/output flags and dependencies, and nodes with buffer names)
    # and log it as a structured trace artifact named after the graph being compiled plus a fused suffix. Side effect: structured trace logging; returns None.
    """
    This is for debugging purposes. It will dump a json file that records graph information.
    The graph can then be used in a simulator: https://fburl.com/code/3l3d3qi4
    """

    class ORMBuffer(TypedDict):
        # [CLASS DOC] HELPER: TypedDict describing a buffer in the exported simulator graph (name, sizes, input/output flags, dependency names).
        name: str
        size_alloc: int
        size_free: int
        size: int  # for backward compatibility
        is_input: bool
        is_output: bool
        deps: list[str]
        unmet_deps: list[str]

    class ORMNode(TypedDict):
        # [CLASS DOC] HELPER: TypedDict describing a node in the exported graph (name and the buffer names it defines).
        name: str
        buffer_names: list[str]

    class ORMGraph(TypedDict):
        # [CLASS DOC] HELPER: TypedDict for the exported graph holding its node and buffer lists.
        nodes: list[ORMNode]
        buffers: list[ORMBuffer]

    orm_buffers: list[ORMBuffer] = []
    orm_nodes: list[ORMNode] = []

    # get orm buffers for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        orm_buf_input_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": input_buf.mpi_buffer.size_free,
            "size_free": input_buf.mpi_buffer.size_free,
            "size": input_buf.mpi_buffer.size_free,
            "is_input": True,
            "is_output": buf_name in graph_outputs,
            "deps": [],
            "unmet_deps": [],
        }
        orm_buffers.append(orm_buf_input_buffer)

    # get orm buffers for scheduler buffers
    name_to_buf: dict[str, SchedulerBuffer] = {
        buf.get_name(): buf for node in nodes for buf in node.get_outputs()
    }  # need to reassign due to probably node pruning
    for buf_name, sched_buf in name_to_buf.items():
        if sched_buf.defining_op is None:
            continue
        deps = [
            pred_buf.get_name()
            for pred_buf in name_to_fused_node[
                sched_buf.defining_op.get_name()
            ].mpi_node.pred_buffers
        ]
        orm_buf_scheduler_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": sched_buf.mpi_buffer.size_alloc,
            "size_free": sched_buf.mpi_buffer.size_free,
            "size": sched_buf.mpi_buffer.size_free,
            "is_input": False,
            "is_output": buf_name in graph_outputs,
            "deps": deps,
            "unmet_deps": [
                buf_name for buf_name in deps if buf_name not in graph_inputs
            ],
        }
        orm_buffers.append(orm_buf_scheduler_buffer)

    # get orm nodes
    for node in nodes:
        orm_node: ORMNode = {
            "name": node.get_name(),
            "buffer_names": list(node.get_buffer_names()),
        }
        orm_nodes.append(orm_node)

    # create the graph object
    g: ORMGraph = {
        "nodes": orm_nodes,
        "buffers": orm_buffers,
    }

    # dump the graph
    import json
    import os

    import torch
    from functorch.compile import get_graph_being_compiled

    name = os.path.splitext(get_graph_being_compiled())[0] + "_fused"

    g_str = json.dumps(g, indent=2)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": "string",
        },
        payload_fn=lambda: g_str,
    )
