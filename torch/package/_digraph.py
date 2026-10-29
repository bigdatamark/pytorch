# mypy: allow-untyped-defs
# [MODULE DOC] Minimal directed graph used by torch.package to track module dependencies.
# Exports the DiGraph class. No import-time side effects.
from collections import deque


class DiGraph:
    # [CLASS DOC] Simple unweighted directed graph with networkx-like method names. State:
    # node attribute mapping, successor and predecessor adjacency mappings (no edge data), and
    # an insertion-order index per node used to find the earliest path. Nodes are any hashable.
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    def __init__(self):
        # [FUNCTION DOC] Creates an empty graph with empty node, successor, predecessor and
        # insertion-order mappings and a zero insertion counter.
        # Dict of node -> dict of arbitrary attributes
        self._node = {}
        # Nested dict of node -> successor node -> nothing.
        # (didn't implement edge data)
        self._succ = {}
        # Nested dict of node -> predecessor node -> nothing.
        self._pred = {}

        # Keep track of the order in which nodes are added to
        # the graph.
        self._node_order = {}
        self._insertion_idx = 0

    def add_node(self, n, **kwargs):
        # [FUNCTION DOC] Adds node n with keyword attributes. A new node gets empty adjacency
        # and the next insertion index; an existing node only has its attributes merged.
        """Add a node to the graph.

        Args:
            n: the node. Can be any object that is a valid dict key.
            **kwargs: any attributes you want to attach to the node.
        """
        if n not in self._node:
            self._node[n] = kwargs
            self._succ[n] = {}
            self._pred[n] = {}
            self._node_order[n] = self._insertion_idx
            self._insertion_idx += 1
        else:
            self._node[n].update(kwargs)

    def add_edge(self, u, v):
        # [FUNCTION DOC] Adds a directed edge from u to v, creating either node if missing.
        # Recording the same edge twice has no further effect.
        """Add an edge to graph between nodes ``u`` and ``v``

        ``u`` and ``v`` will be created if they do not already exist.
        """
        # add nodes
        self.add_node(u)
        self.add_node(v)

        # add the edge
        self._succ[u][v] = True
        self._pred[v][u] = True

    def successors(self, n):
        # [FUNCTION DOC] Returns an iterator over nodes with an edge from n. Raises an error
        # (ValueError) if n is not in the graph.
        """Returns an iterator over successor nodes of n."""
        try:
            return iter(self._succ[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    def predecessors(self, n):
        # [FUNCTION DOC] Returns an iterator over nodes with an edge into n. Raises an error
        # (ValueError) if n is not in the graph.
        """Returns an iterator over predecessor nodes of n."""
        try:
            return iter(self._pred[n])
        except KeyError as e:
            raise ValueError(f"The node {n} is not in the digraph.") from e

    @property
    def edges(self):
        # [FUNCTION DOC] Property generating every edge as a (source, target) pair, grouped by
        # source node in insertion order.
        """Returns an iterator over all edges (u, v) in the graph"""
        for n, successors in self._succ.items():
            for succ in successors:
                yield n, succ

    @property
    def nodes(self):
        # [FUNCTION DOC] Property returning the live mapping of node to its attribute mapping
        # (not a copy).
        """Returns a dictionary of all nodes to their attributes."""
        return self._node

    def __iter__(self):
        # [FUNCTION DOC] Returns an iterator over the nodes in insertion order.
        """Iterate over the nodes."""
        return iter(self._node)

    def __contains__(self, n):
        # [FUNCTION DOC] Membership test for a node; unhashable values yield false instead of
        # raising.
        """Returns True if ``n`` is a node in the graph, False otherwise."""
        try:
            return n in self._node
        except TypeError:
            return False

    def forward_transitive_closure(self, src: str) -> set[str]:
        # [FUNCTION DOC] Breadth-first search along successor edges; returns the set of nodes
        # reachable from src, including the start.
        # KNOWN ISSUE: the set and queue are built directly from src, so a multi-character
        # string is split into one start node per character; only correct for one-char names.
        """Returns a set of nodes that are reachable from src"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.successors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def backward_transitive_closure(self, src: str) -> set[str]:
        # [FUNCTION DOC] Breadth-first search along predecessor edges; returns the set of nodes
        # that can reach src, including the start.
        # KNOWN ISSUE: same per-character splitting of a multi-character src as the forward
        # closure.
        """Returns a set of nodes that are reachable from src in reverse direction"""

        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def all_paths(self, src: str, dst: str):
        # [FUNCTION DOC] Builds the subgraph of nodes and edges lying on some path from src to
        # dst: computes everything reachable from src, then walks predecessors back from dst
        # keeping only reachable nodes. Returns the dot text of that subgraph; if dst is not
        # reachable, returns an empty graph object instead.
        # KNOWN ISSUE: inconsistent return type (graph object vs dot string), and the
        # annotation is missing; also the walk has no visited set, so cyclic or diamond-shaped
        # graphs revisit nodes (possible non-termination on cycles); dst is also split per
        # character when used to seed the queue.
        """Returns a subgraph rooted at src that shows all the paths to dst."""

        result_graph = DiGraph()
        # First compute forward transitive closure of src (all things reachable from src).
        forward_reachable_from_src = self.forward_transitive_closure(src)

        if dst not in forward_reachable_from_src:
            return result_graph

        # Second walk the reverse dependencies of dst, adding each node to
        # the output graph iff it is also present in forward_reachable_from_src.
        # we don't use backward_transitive_closures for optimization purposes
        working_set = deque(dst)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n in forward_reachable_from_src:
                    result_graph.add_edge(n, cur)
                    # only explore further if its reachable from src
                    working_set.append(n)

        return result_graph.to_dot()

    def first_path(self, dst: str) -> list[str]:
        # [FUNCTION DOC] Walks back from dst, at each step moving to the predecessor that was
        # added to the graph earliest, until a node has no predecessor. Returns the nodes in
        # order from the root to dst. Stops early if a predecessor has no recorded order.
        """Returns a list of nodes that show the first path that resulted in dst being added to the graph."""
        path = []

        while dst:
            path.append(dst)
            candidates = self._pred[dst].keys()
            dst, min_idx = "", None
            for candidate in candidates:
                idx = self._node_order.get(candidate, None)
                if idx is None:
                    break
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                    dst = candidate

        return list(reversed(path))

    def to_dot(self) -> str:
        # [FUNCTION DOC] Returns Graphviz text for the graph: left-to-right layout, box nodes,
        # one quoted edge line per edge.
        """Returns the dot representation of the graph.

        Returns:
            A dot representation of the graph.
        """
        edges = "\n".join(f'"{f}" -> "{t}";' for f, t in self.edges)
        return f"""\
digraph G {{
rankdir = LR;
node [shape=box];
{edges}
}}
"""
