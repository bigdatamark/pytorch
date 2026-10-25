# [MODULE DOC]
# Identifies groups of identical, non-overlapping subgraph regions in an FX graph (used for
# deduplicating repeated code into subgraph higher-order ops). Nodes are hashed at insertion time
# by source location, instruction pointer, global torch state, and normalized argument metadata;
# equal hashes form duplicate groups, which are then grown backward through argument edges into
# maximal identical regions. Contains the pickler used for hashing, a backward BFS iterator, a
# per-region wrapper tracking ancestors for cycle checks, and the GraphRegionTracker.
# Import-time side effects: creates a module logger and the "graph_region_expansion" artifact logger.
"""
This module provides functionality for tracking and managing regions in computational graphs.
It supports graph optimization by identifying and grouping similar regions based on their
structure and behavior. The module implements algorithms for:

1. Tracking nodes and their relationships in the computational graph
2. Identifying identical or similar regions across the graph
3. Managing graph regions for optimization purposes
4. Supporting deduplication and other graph transformation passes

The core functionality revolves around the GraphRegionTracker class which maintains
mappings between nodes and their duplicates, enabling efficient graph analysis and
optimization operations.
"""

from __future__ import annotations

import copyreg
import io
import logging
import math
import operator
import pickle
from collections import defaultdict, deque
from dataclasses import fields
from typing import Any, TYPE_CHECKING, TypeVar

import torch._logging
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten

from .graph_utils import _get_flat_args_unique


T = TypeVar("T")


if TYPE_CHECKING:
    from collections.abc import Callable

    from .symbolic_convert import InstructionTranslatorBase


Node = torch.fx.Node
Region = list[Node]
IdenticalNodes = list[Node]
GlobalStateKey = tuple[
    bool,
    bool,
    int,
    tuple[bool, bool],
    tuple[bool, bool],
    torch.dtype,
    bool,
    bool,
    bool,
    bool,
]

log = logging.getLogger(__name__)
graph_expansion_log = torch._logging.getArtifactLogger(
    __name__, "graph_region_expansion"
)


def debug_log(msg: str, *args) -> None:  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Emit a debug message with lazy formatting arguments to the graph region expansion artifact logger.
    graph_expansion_log.debug(msg, *args)


def _extract_tensor_metadata_for_node_hash(
    x: torch.Tensor,
) -> tuple[Callable[[T], T], tuple[Any, ...]]:
    # [FUNCTION DOC] HELPER: Pickle reducer for fake tensors. Extracts the inductor cache-key tensor
    # metadata and returns an identity-callable plus a tuple of its field values, so tensors with
    # equal metadata pickle identically. Imports inductor codecache lazily.
    from torch._inductor.codecache import _ident, extract_tensor_metadata_for_cache_key

    out = []
    metadata = extract_tensor_metadata_for_cache_key(x)
    for field in fields(metadata):
        out.append(getattr(metadata, field.name))

    return (_ident, tuple(out))


class NodeHashException(Exception):
    # [CLASS DOC] Raised when a node's hash key cannot be pickled; callers skip tracking that node.
    pass


class InputPickler(pickle.Pickler):
    # [CLASS DOC] HELPER: Pickler that serializes node hash keys deterministically into a reusable
    # in-memory buffer. Its dispatch table is the default copyreg table extended so fake tensors
    # pickle as their cache-key metadata and symbolic ints/bools/floats pickle as their string form.
    # Fast mode is on (no memoization). The buffer is cleared after every dumps call.
    def __init__(self) -> None:
        # [FUNCTION DOC] Create the byte buffer, initialize the base pickler on it, and install the
        # custom dispatch table and fast mode.
        from torch._inductor.codecache import _ident

        stream = io.BytesIO()
        self._stream = stream
        super().__init__(stream)
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(
            {
                FakeTensor: _extract_tensor_metadata_for_node_hash,
                torch.SymInt: lambda x: (_ident, (str(x),)),
                torch.SymBool: lambda x: (_ident, (str(x),)),
                torch.SymFloat: lambda x: (_ident, (str(x),)),
            }
        )
        self.fast = True

    def dumps(self, obj: Any) -> bytes:
        # [FUNCTION DOC] Serialize obj and return the bytes. Raises NodeHashException if pickling hits a
        # type or attribute error. Always resets the shared buffer afterwards.
        """
        Pickle an object and return a byte string.
        """
        try:
            self.dump(obj)
            return self._stream.getvalue()
        except (TypeError, AttributeError) as e:
            raise NodeHashException from e
        finally:
            self._stream.seek(0)
            self._stream.truncate(0)


def _extract_args(arg: Any) -> Any:
    # [FUNCTION DOC] HELPER: Map an argument to a hashable stand-in: for a graph node, its recorded
    # example value (or null); tensors and ints are returned as is; anything else becomes null.
    if isinstance(arg, Node):
        return arg.meta.get("example_value")
    elif isinstance(arg, (torch.Tensor, int)):
        return arg
    else:
        return None


def _normalize_args(
    node: Node,
) -> tuple[tuple[str, ...], tuple[Any | None, ...]]:
    # [FUNCTION DOC] HELPER: Flatten a node's positional args and key-sorted kwargs; return the sorted
    # keyword names plus a tuple of stand-in values for every flattened argument. Used in node hashing.
    flat_args, _ = tree_flatten(node.args)
    sorted_kwargs = sorted(node.kwargs.items(), key=operator.itemgetter(0))
    sorted_keys = tuple(sorted(node.kwargs.keys()))
    flat_kwargs, _ = tree_flatten(sorted_kwargs)
    all_args = flat_args + flat_kwargs
    return (sorted_keys, tuple(_extract_args(arg) for arg in all_args))


def _sort_with_ref_region(
    index_to_rank: dict[int, int], regions: list[list[Any]]
) -> None:
    # [FUNCTION DOC] HELPER: Reorder every region in place using the order implied by the first
    # region: positions are sorted by the rank given in index_to_rank, and the same permutation is
    # applied to all regions so they stay aligned. Mutates the regions; returns nothing.
    # sort topologically
    # we need to handle edge cases where some nodes have no dependencies
    # so first we map each node to its ranking
    ref_region = regions[0]
    sorted_indices = sorted(range(len(ref_region)), key=lambda i: index_to_rank[i])
    for region in regions:
        region[:] = [region[i] for i in sorted_indices]


def get_global_state_key() -> GlobalStateKey:
    # [FUNCTION DOC] Return a tuple snapshot of global torch state that affects op semantics: grad and
    # inference mode, thread count, cublas reduced-precision and tf32 flags, default dtype,
    # deterministic-algorithm flags, and saved-tensor-hooks enablement. Read-only.
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.get_num_threads(),
        torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
        torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        torch.get_default_dtype(),
        torch.are_deterministic_algorithms_enabled(),
        torch._C._get_cublas_allow_tf32(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch._C._autograd._saved_tensors_hooks_is_enabled(),  # type: ignore[attr-defined]
    )


# This is typical BFS with the caveat
# that a node's children need to be explicitly
# added with the add_children() method
# The flow is yield a node and check if it's valid for all regions
# if not valid, discard and continue onto the next node
# Note: this iterates backward through the graph by looking at args/kwargs
# of a node
class BackwardBfsArgIter:
    # [CLASS DOC] HELPER: Breadth-first iterator walking backward from an origin node through node
    # arguments. State: the current candidate and a queue of pending nodes. Children of a node are
    # enqueued only when the caller explicitly calls add_children, so callers control which branches
    # are explored. Use create to build one with the origin already consumed.
    def __init__(self, origin: Node) -> None:
        # [FUNCTION DOC] Start with origin as the current node and an empty queue.
        self._cur: Node | None = origin
        self._queue: deque[Node | None] = deque()

    @staticmethod
    def create(origin: Node) -> BackwardBfsArgIter:
        # [FUNCTION DOC] Build an iterator for origin, enqueue its argument nodes, and consume the origin
        # itself. Raises an error if the origin could not be consumed.
        it = BackwardBfsArgIter(origin)
        it.add_children(origin)
        # pop the origin node, since it is the origin of
        # the region and does not need to be considered for addition
        if not it.next():
            raise AssertionError("expected origin node to be popped from iterator")
        return it

    def next(self) -> Node | None:
        # [FUNCTION DOC] Return the current node and advance to the next queued node (or none if the
        # queue is empty). Returns none when exhausted.
        ret = self._cur
        if not self._queue:
            self._cur = None
        else:
            self._cur = self._queue.popleft()
        return ret

    def peek(self) -> Node | None:
        # [FUNCTION DOC] Return the current node without advancing.
        return self._cur

    def add_children(self, node: Node) -> None:
        # [FUNCTION DOC] Enqueue each distinct node appearing in the node's args and kwargs.
        flat_args = _get_flat_args_unique(node, {})
        for arg in flat_args:
            if isinstance(arg, Node):
                self._append(arg)

    def _append(self, arg: Node) -> None:
        # [FUNCTION DOC] HELPER: Make arg the current node if none is current, else add to the queue.
        # KNOWN ISSUE: nodes are not deduplicated across calls, so a node reachable through several
        # paths can be yielded more than once; callers rely on the seen-node checks to cope.
        if self._cur is None:
            self._cur = arg
        else:
            self._queue.append(arg)

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string with the current node and queue.
        return f"BackwardBfsArgIter(cur={self._cur}, queue={self._queue})"


class GraphRegionTracker:
    # [CLASS DOC] Records nodes added to the output graph, grouping those with equal hash keys, and later
    # computes maximal identical regions. State: hash_to_duplicates (hash to node list),
    # node_to_duplicates (node to its group list, shared by identity), node_to_mutated_arg_positions
    # (node to flattened argument positions mutated), and an InputPickler. Key methods: track_node,
    # track_node_mutations, add_node_mutation, get_identical_regions.
    """
    GraphRegionTracker tracks each node added to the output graph and generates a key based on the source location,
    instruction pointer, input shapes, and global state at the time the node is inserted into the graph. Nodes with
    the same key are grouped together in a list of identical nodes (the value of node_to_duplicates).

    hash_to_duplicates: Dict[str, IdenticalNodes] - A dictionary mapping the key to a list of identical nodes
    node_to_duplicates: Dict[Node, IdenticalNodes] - A dictionary mapping a node to the list of identical nodes it belongs to
    input_pickler: InputPickler - An instance of InputPickler used to generate a node hash
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize empty duplicate-group mappings, mutation-position mapping, and a pickler.
        self.hash_to_duplicates: dict[str, IdenticalNodes] = defaultdict(list)
        self.node_to_duplicates: dict[Node, IdenticalNodes] = {}
        # Note: position is in flattened args/kwargs list
        self.node_to_mutated_arg_positions: dict[Node, OrderedSet[int]] = {}
        self.input_pickler = InputPickler()

    def _hash_node(
        self, filename: str, lineno: int, instruction_pointer: int | None, node: Node
    ) -> str:
        # [FUNCTION DOC] HELPER: Compute a SHA-256 hex digest of a key made from global torch state, file
        # name, line number, instruction pointer, and the node's normalized args. Raises
        # NodeHashException if the key cannot be pickled.
        from torch._inductor.codecache import sha256_hash

        key = (
            get_global_state_key(),
            filename,
            lineno,
            instruction_pointer,
            _normalize_args(node),
        )
        return sha256_hash(self.input_pickler.dumps(key))

    def _is_identical(self, n0: Node, n1: Node) -> bool:
        # [FUNCTION DOC] HELPER: True if both nodes are tracked, are distinct, and share the very same
        # duplicate group.
        return (
            n0 in self.node_to_duplicates
            and n1 in self.node_to_duplicates
            and self.node_to_duplicates[n0] is self.node_to_duplicates[n1]
            and n0 is not n1
        )

    def track_node(self, tx: InstructionTranslatorBase, node: Node) -> None:
        # [FUNCTION DOC] Hash the node using the translator's code file name, line number and instruction
        # pointer, and append it to the matching duplicate group. Nodes already tracked are ignored.
        # If hashing fails, a debug message is logged and the node is left untracked (no raise).
        """
        The main entry point for tracking a node. This function will hash the node argument and group
        nodes with the same hash together. It updates the hash_to_duplicates and node_to_duplicates dictionaries
        to track the new node.
        """
        try:
            if (
                node not in self.node_to_duplicates
            ):  # don't allow nodes to be added twice
                duplicates = self.hash_to_duplicates[
                    self._hash_node(
                        tx.f_code.co_filename, tx.lineno, tx.instruction_pointer, node
                    )
                ]
                duplicates.append(node)
                self.node_to_duplicates[node] = duplicates
        except NodeHashException as e:
            log.debug("Unable to hash node %s with exception %s", node, e)

    def track_node_mutations(
        self,
        node: Node,
        flat_args_kwargs: list[Any],
        id_to_initial_version: dict[int, int],
    ) -> None:
        # [FUNCTION DOC] For each flattened arg whose object identity is in id_to_initial_version and whose
        # current version counter differs from the recorded one, record its position as mutated by node.
        # Stores nothing if no position was mutated. Mutates node_to_mutated_arg_positions.
        """
        This function tracks which argument positions are mutated by the given node. Subgraph HOP does not support
        input mutations today so we will skip regions which have inputs that are mutated.
        """
        mutated_arg_positions = OrderedSet[int]()
        for i, arg in enumerate(flat_args_kwargs):
            val_id = id(arg)
            if (
                val_id in id_to_initial_version
                and id_to_initial_version[val_id] != arg._version
            ):
                mutated_arg_positions.add(i)

        if mutated_arg_positions:
            self.node_to_mutated_arg_positions[node] = mutated_arg_positions

    def add_node_mutation(
        self,
        node: Node,
        arg_pos: int,
    ) -> None:
        # [FUNCTION DOC] Record that the flattened argument at arg_pos of node is mutated, creating the
        # node's position set if needed.
        if node in self.node_to_mutated_arg_positions:
            self.node_to_mutated_arg_positions[node].add(arg_pos)
        else:
            self.node_to_mutated_arg_positions[node] = OrderedSet([arg_pos])

    def get_identical_regions(self, graph: torch.fx.Graph) -> list[list[Region]]:
        # [FUNCTION DOC] Build region groups from duplicate groups with more than one member present in the
        # graph (each region initially one node), order groups so those latest in the graph go first,
        # expand each group backward as far as all regions stay identical and non-overlapping, and sort
        # each group's regions topologically using the first region as reference. Returns the groups
        # whose regions have more than one node. Assumes tracked nodes belong to graph. Mutates
        # nothing outside local state.
        """
        This function is responsible for extracting the largest regions of identical nodes from the given graph.
        **Note**: This function assumes the nodes that have been tracked with track_node are in the provided graph argument.

        The algorithm proceeds as follows:
        The nodes tracked via track_node above are organized into region groups. The initial region groups look like this:
        [[IdenticalNode1], [IdenticalNode2], [IdenticalNode3]] and each sublist is called a region. For each region group
        (starting at the topologically latest region group), the inner regions are gradually expanded one node at time from
        the flattened args and kwargs of the node in each region provided that for all regions in the group, the nodes being
        added are also identical (ie have the same key computed by track_node). This is checked by verifying that the two
        nodes have the same identical node list in node_to_duplicates.
        """
        topological_ranking = {node: i for i, node in enumerate(graph.nodes)}
        region_groups_with_rank = []
        # needed to detect if replacing a region will create cycles
        node_to_recursive_ancestors = _populate_recursive_ancestor_map(graph)

        # Create region groups; a region group is a group
        # of regions that are all identical. In this initial state
        # each region in the group is a single node, and we discard
        # groups that are only a single region.
        # We track the topological ranking to start with groups later in the graph
        # the reason for this is that we will necessarily create the largest groups first.
        for group in self.hash_to_duplicates.values():
            if len(group) > 1:
                # pyrefly: ignore [implicit-any]
                region_group = []
                min_rank = math.inf

                for node in group:
                    # some nodes aren't in the topo ranking?
                    if node in topological_ranking:
                        min_rank = min(min_rank, topological_ranking[node])
                        region_group.append([node])

                if len(region_group) > 1:
                    region_groups_with_rank.append((region_group, min_rank))

        region_groups_with_rank.sort(key=lambda rg: -rg[1])
        region_groups = [rg for rg, _ in region_groups_with_rank]

        # We start from regions later in the graph and expand them earlier
        # as a result, we will create the largest regions first and they won't
        # overlap.
        seen_nodes: set[Node] = set()
        for region_group in region_groups:
            fully_expand_region_group(
                region_group,
                seen_nodes,
                node_to_recursive_ancestors,
                self._is_identical,
            )
            # sort topologically
            # we need to handle edge cases where some nodes have no dependencies
            # so first we map each node to its ranking,
            ref_region = region_group[0]
            index_to_rank = {
                index: topological_ranking[n] for index, n in enumerate(ref_region)
            }
            _sort_with_ref_region(index_to_rank, region_group)

        return [
            region_group for region_group in region_groups if len(region_group[0]) > 1
        ]

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string of the duplicate mappings.
        return f"GraphRegionTracker(hash_to_duplicates={self.hash_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"


class RegionWrapper:
    # [CLASS DOC] HELPER: Wraps one region under expansion. State: the region list (mutated in place),
    # a backward BFS iterator of candidate nodes, the ordered set of member nodes, the union of all
    # members' recursive ancestors, and the shared ancestor map. Provides cycle checking before a
    # node is added.
    """Holds state for regions e.g. ancestors and new candidate nodes for consideration"""

    def __init__(
        self, region: Region, node_to_recursive_ancestors: dict[Node, set[Node]]
    ) -> None:
        # [FUNCTION DOC] Initialize from a single-node region: create the BFS iterator, member set, and
        # ancestor set. Raises an error if the region does not have exactly one node.
        if len(region) != 1:
            raise AssertionError("all regions should start with one node")
        node = region[0]
        self.node_to_recursive_ancestors = node_to_recursive_ancestors
        self.iter = BackwardBfsArgIter.create(node)
        self.nodes_unique = OrderedSet([node])
        self.ancestors = set(node_to_recursive_ancestors[node])
        self.region = region

    def next_candidate(self) -> Node | None:
        # [FUNCTION DOC] Pop and return the next candidate node from the iterator, or none if exhausted.
        return self.iter.next()

    def will_inclusion_create_cycle(self, node: Node) -> bool:
        # [FUNCTION DOC] Return True if any user of node outside the region is an ancestor of the region,
        # meaning that collapsing the node into the region would create a dependency cycle.
        external_users = [user for user in node.users if user not in self.nodes_unique]
        for user in external_users:
            if user in self.ancestors:
                return True

        return False

    def add(self, node: Node) -> None:
        # [FUNCTION DOC] Add node to the region: record membership, append to the region list, enqueue its
        # argument nodes as future candidates, and merge its ancestors into the region's ancestor set.
        self.nodes_unique.add(node)
        self.region.append(node)
        self.iter.add_children(node)
        self.ancestors.update(self.node_to_recursive_ancestors[node])


def fully_expand_region_group(
    regions: list[Region],
    seen_nodes: set[Node],
    node_to_recursive_ancestors: dict[Node, set[Node]],
    is_identical_fn: Callable[[Node, Node], bool],
) -> None:
    # [FUNCTION DOC] Grow every single-node region in the group in lockstep, in place. Each step takes the
    # next backward candidate from each region and adds the set only if all candidates exist, are
    # unclaimed, distinct, not placeholders or attribute fetches, pairwise identical to the first
    # region's candidate, and would not create a cycle. Added nodes are put in seen_nodes. At the end each
    # region is reversed into topological order. Raises an error if any region does not start with one
    # node or if the number of nodes to add differs from the number of regions.
    debug_log("--------------------------------------------------")
    debug_log("expanding new region group: %s", regions)

    # All regions should start with 1 node
    if not all(len(region) == 1 for region in regions):
        raise AssertionError("all regions should start with one node")
    region_wrappers = [
        RegionWrapper(region, node_to_recursive_ancestors) for region in regions
    ]

    nodes_to_add = OrderedSet[Node]()
    current_node = region_wrappers[0].next_candidate()

    # No children
    if current_node is None:
        return

    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
    while current_node:
        add_to_all_regions = not region_wrappers[0].will_inclusion_create_cycle(
            current_node
        )
        nodes_to_add.clear()
        nodes_to_add.add(current_node)
        for region_wrapper in region_wrappers[1:]:
            candidate = region_wrapper.next_candidate()

            debug_log("--------------------")
            debug_log(
                "considering candidate: %s, cur_node: %s", candidate, current_node
            )

            if not candidate or not add_to_all_regions:
                add_to_all_regions = False
                continue

            debug_log(
                "candidate in previously claimed nodes?: %s", candidate in seen_nodes
            )
            debug_log("is_identical: %s", is_identical_fn(candidate, current_node))

            add_to_all_regions &= (
                candidate not in seen_nodes
                and candidate not in nodes_to_add
                and candidate.op != "placeholder"
                and candidate.op != "get_attr"
                and is_identical_fn(candidate, current_node)
                and not region_wrapper.will_inclusion_create_cycle(candidate)
            )
            nodes_to_add.add(candidate)

            debug_log(f"add_to_all_regions: {add_to_all_regions}")
            debug_log("--------------------")

        if add_to_all_regions:
            if len(region_wrappers) != len(nodes_to_add):
                raise AssertionError(
                    "Number of nodes to add must equal the number of regions"
                )
            for region_wrapper, node in zip(region_wrappers, nodes_to_add):
                region_wrapper.add(node)
                debug_log("adding %s's children", node)
                debug_log("%s %s", node.args, list(node.kwargs.items()))
                seen_nodes.add(node)

        current_node = region_wrappers[0].next_candidate()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()

    debug_log("end expand new region group: %s", regions)
    debug_log("--------------------------------------------------")


def _populate_recursive_ancestor_map(graph: torch.fx.Graph) -> dict[Node, set[Node]]:
    # [FUNCTION DOC] HELPER: Return a mapping from each graph node to the set of all nodes it transitively
    # depends on, computed in one pass over the graph in order (relies on topological node order).
    node_to_recursive_ancestors: dict[Node, set[Node]] = {}
    for node in graph.nodes:
        node_to_recursive_ancestors[node] = set()
    for node in graph.nodes:
        all_args = _get_flat_args_unique(node, {})
        for arg in all_args:
            if isinstance(arg, Node):
                node_to_recursive_ancestors[node].update(
                    node_to_recursive_ancestors[arg]
                )
                node_to_recursive_ancestors[node].add(arg)
    return node_to_recursive_ancestors
