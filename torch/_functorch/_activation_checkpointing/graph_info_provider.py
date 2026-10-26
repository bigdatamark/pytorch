# [MODULE DOC] Defines GraphInfoProvider, a container for joint-graph structure (node order, edges) and
# per-node runtime and memory data for activation checkpointing candidates. It lazily builds
# networkx directed graphs and a simplified FX graph used by knapsack-based selection.
# Import-time side effects: imports networkx.
from typing import Any

import networkx as nx

from torch.fx import Graph, Node


class GraphInfoProvider:
    # [CLASS DOC] Holds node names in order, edges, the ordered list of recomputable (banned) nodes, and
    # dictionaries of runtimes and memories per node. Four derived graphs are built lazily and
    # cached by name in a private dictionary. Knapsack input lists may be recorded or derived.
    """
    This class provides information about the graph, such as the nodes, edges, and their runtime and memory requirements.
    It also provides methods to create graphs from the information provided.
    """

    __RECOMPUTABLE_NODE_ONLY_GRAPH = "recomputable_node_only_graph"
    __RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT = (
        "recomputable_node_only_graph_with_larger_graph_context"
    )
    __FULL_NX_JOINT_GRAPH = "full_nx_joint_graph"
    __SIMPLIFIED_FX_JOINT_GRAPH = "fx_joint_graph"

    def __init__(
        self,
        graph_nodes_in_order: list[str],
        graph_edges: list[tuple[str, str]],
        all_recomputable_banned_nodes: list[str],
        all_node_runtimes: dict[str, float] | None = None,
        all_node_memories: dict[str, float] | None = None,
        recorded_knapsack_input_memories: list[float] | None = None,
        recorded_knapsack_input_runtimes: list[float] | None = None,
        joint_graph: Graph | None = None,
    ) -> None:
        # [FUNCTION DOC] Store graph structure and per-node data. Runtimes and memories come from the given
        # mappings, or, if absent, from the recorded knapsack input lists paired positionally with
        # all_recomputable_banned_nodes. Raises ValueError if neither source is given for
        # runtimes or for memories. Lazy graph caches start empty.
        # KNOWN ISSUE: the joint_graph argument is accepted but never used or stored.
        self.graph_nodes_in_order = graph_nodes_in_order
        self.graph_edges = graph_edges
        self.all_node_runtimes: dict[str, float] = dict()
        if all_node_runtimes is None:
            if recorded_knapsack_input_runtimes is None:
                raise ValueError(
                    "Either all_node_runtimes or recorded_knapsack_input_runtimes must be provided."
                )
            self.all_node_runtimes = {
                node: recorded_knapsack_input_runtimes[i]
                for i, node in enumerate(all_recomputable_banned_nodes)
            }
        else:
            self.all_node_runtimes.update(all_node_runtimes)
        self.all_node_memories: dict[str, float] = dict()
        if all_node_memories is None:
            if recorded_knapsack_input_memories is None:
                raise ValueError(
                    "Either all_node_memories or recorded_knapsack_input_memories must be provided."
                )
            self.all_node_memories = {
                node: recorded_knapsack_input_memories[i]
                for i, node in enumerate(all_recomputable_banned_nodes)
            }
        else:
            self.all_node_memories.update(all_node_memories)
        self.all_recomputable_banned_nodes = all_recomputable_banned_nodes
        self.all_recomputable_banned_nodes_set = set(all_recomputable_banned_nodes)
        self.recorded_knapsack_input_memories = recorded_knapsack_input_memories
        self.recorded_knapsack_input_runtimes = recorded_knapsack_input_runtimes
        self._lazily_initialized_graphs: dict[str, Any] = {
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH: None,
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT: None,
            self.__FULL_NX_JOINT_GRAPH: None,
            self.__SIMPLIFIED_FX_JOINT_GRAPH: None,
        }

    @classmethod
    def inialize_from_graph(
        cls,
        joint_graph: Graph,
        all_recomputable_banned_nodes: list[Node],
        recorded_knapsack_input_memories: list[float],
        recorded_knapsack_input_runtimes: list[float],
    ) -> "GraphInfoProvider":
        # [FUNCTION DOC] Alternate constructor: derive node names (in order) and (node, user) edges from an FX
        # joint graph, convert banned nodes to names, and build a provider using the recorded
        # knapsack memory and runtime lists. Returns the new provider.
        # KNOWN ISSUE: method name is misspelled (inialize).
        """
        Enables initialization from a joint graph.
        """
        graph_nodes_in_order = [node.name for node in joint_graph.nodes]
        graph_edges = [
            (node.name, user.name) for node in joint_graph.nodes for user in node.users
        ]
        all_recomputable_banned_node_names = [
            node.name for node in all_recomputable_banned_nodes
        ]
        return cls(
            graph_nodes_in_order=graph_nodes_in_order,
            graph_edges=graph_edges,
            all_recomputable_banned_nodes=all_recomputable_banned_node_names,
            recorded_knapsack_input_memories=recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=recorded_knapsack_input_runtimes,
            joint_graph=joint_graph,
        )

    @property
    def recomputable_node_only_graph(self) -> nx.DiGraph:
        # [FUNCTION DOC] Lazily build and cache the graph containing only recomputable nodes and the edges among them.
        if self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH] is None:
            self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH] = (
                self._create_recomputable_node_only_graph()
            )
        return self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH]

    @property
    def recomputable_node_only_graph_with_larger_graph_context(self) -> nx.DiGraph:
        # [FUNCTION DOC] Lazily build and cache the recomputable-node graph whose edges reflect reachability
        # through the full joint graph (see the private builder).
        if (
            self._lazily_initialized_graphs[
                self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
            ]
            is None
        ):
            self._lazily_initialized_graphs[
                self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
            ] = self._create_recomputable_node_only_graph_with_larger_graph_context()
        return self._lazily_initialized_graphs[
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
        ]

    @property
    def full_joint_nx_graph(self) -> nx.DiGraph:
        # [FUNCTION DOC] Lazily build and cache the full joint graph as a networkx digraph without output nodes.
        if self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH] is None:
            self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH] = (
                self._create_full_joint_graph()
            )
        return self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH]

    @property
    def simplified_fx_joint_graph(self) -> Graph:
        # [FUNCTION DOC] Lazily build and cache a placeholder-and-identity FX graph with the same dependency structure.
        if self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH] is None:
            self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH] = (
                self._recreate_psuedo_joint_graph()
            )
        return self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH]

    def get_non_ac_peak_memory(self) -> float:
        # [FUNCTION DOC] Return the sum of memory over all recomputable candidate nodes (the cost of saving all of them).
        return sum(
            self.all_node_memories[node_name]
            for node_name in self.all_recomputable_banned_nodes_set
        )

    def get_theoretical_max_runtime(self) -> float:
        # [FUNCTION DOC] Return the sum of runtime over all recomputable candidate nodes (the cost of recomputing all of them).
        return sum(
            self.all_node_runtimes[node_name]
            for node_name in self.all_recomputable_banned_nodes_set
        )

    def get_knapsack_memory_input(self) -> list[float]:
        # [FUNCTION DOC] Return the recorded knapsack memory list if non-empty, else memories of the candidates in banned-node order.
        return (
            self.recorded_knapsack_input_memories
            if self.recorded_knapsack_input_memories
            else [
                self.all_node_memories[node_name]
                for node_name in self.all_recomputable_banned_nodes
            ]
        )

    def get_knapsack_runtime_input(self) -> list[float]:
        # [FUNCTION DOC] Return the recorded knapsack runtime list if non-empty, else runtimes of the candidates in banned-node order.
        return (
            self.recorded_knapsack_input_runtimes
            if self.recorded_knapsack_input_runtimes
            else [
                self.all_node_runtimes[node_name]
                for node_name in self.all_recomputable_banned_nodes
            ]
        )

    def _create_recomputable_node_only_graph(self) -> nx.DiGraph:
        # [FUNCTION DOC] HELPER: Build a digraph with one node per candidate and an edge for each original edge whose endpoints are both candidates.
        graph = nx.DiGraph()
        for recomputable_node in self.all_recomputable_banned_nodes:
            graph.add_node(recomputable_node)

        for a, b in self.graph_edges:
            if (
                a in self.all_recomputable_banned_nodes_set
                and b in self.all_recomputable_banned_nodes_set
            ):
                graph.add_edge(a, b)
        return graph

    def _create_recomputable_node_only_graph_with_larger_graph_context(
        self,
    ) -> nx.DiGraph:
        # [FUNCTION DOC] HELPER: Build a digraph over candidates. For each candidate, find the candidates
        # reachable from it in the full joint graph (breadth-first); add an edge to a reachable
        # candidate only if it is not also reachable through another reachable candidate
        # (a transitive reduction restricted to candidates).
        # Create a dictionary to store the reachable nodes for each node
        all_recomputable_banned_nodes_set = set(self.all_recomputable_banned_nodes)

        reachable_nodes = {}
        for node in all_recomputable_banned_nodes_set:
            # Use BFS to find all reachable nodes
            predecessors = dict(nx.bfs_predecessors(self.full_joint_nx_graph, node))
            reachable_recomputable_nodes = set(predecessors.keys()).intersection(
                all_recomputable_banned_nodes_set
            )
            reachable_nodes[node] = reachable_recomputable_nodes
        # Create the candidate graph
        candidate_graph = nx.DiGraph()
        candidate_graph.add_nodes_from(all_recomputable_banned_nodes_set)
        for node1 in all_recomputable_banned_nodes_set:
            for node2 in reachable_nodes[node1]:
                # Check if there is an overlapping path
                overlapping_path = False
                for intermediate_node in reachable_nodes[node1]:
                    if (
                        intermediate_node != node2
                        and node2 in reachable_nodes[intermediate_node]
                    ):
                        overlapping_path = True
                        break
                if not overlapping_path:
                    candidate_graph.add_edge(node1, node2)
        return candidate_graph

    def _create_full_joint_graph(self) -> nx.DiGraph:
        # [FUNCTION DOC] HELPER: Build a digraph of every node and edge, skipping nodes named output and their edges.
        graph = nx.DiGraph()
        for node in self.graph_nodes_in_order:
            if node == "output":
                continue
            graph.add_node(node)

        for a, b in self.graph_edges:
            if a == "output" or b == "output":
                continue
            graph.add_edge(a, b)
        return graph

    def _recreate_psuedo_joint_graph(self) -> Graph:
        # [FUNCTION DOC] HELPER: Rebuild an FX graph from names and edges: nodes without inputs become
        # placeholders, others become identity-tuple calls over their input nodes, in the stored
        # order. Raises ValueError if an edge references an unknown node.
        # Create a dictionary to store the dependencies of each node
        node_dependencies: dict[str, list[str]] = {
            node: [] for node in self.graph_nodes_in_order
        }
        for a, b in self.graph_edges:
            if a not in node_dependencies or b not in node_dependencies:
                raise ValueError(f"Edge ({a}, {b}) references a non-existent node.")
            node_dependencies[b].append(a)

        joint_graph = Graph()
        # Create nodes in the graph
        nodes: dict[str, Node] = {}
        for node_name in self.graph_nodes_in_order:
            input_nodes = [nodes[dep] for dep in node_dependencies[node_name]]
            if input_nodes:
                node = joint_graph.call_function(lambda *x: x, tuple(input_nodes))
                node.name = node_name
            else:
                node = joint_graph.placeholder(node_name)
            nodes[node_name] = node
        return joint_graph

    def _visualize_recomputable_candidate_graph_with_larger_context(
        self,
        layout_k: float = 0.5,
        layout_iterations: int = 30,
    ) -> None:
        # [FUNCTION DOC] HELPER: Debug helper that draws the larger-context candidate graph with matplotlib
        # (spring layout, nodes colored by memory, labeled by index) and shows the window.
        # Requires matplotlib and a display; blocks on the plot.
        # KNOWN ISSUE: reads node attributes memory and index that the graph builder never sets,
        # so the memory lookup raises KeyError; the label falls back to the node name.
        """
        Visualize the recomputable candidate graph with larger context.
        """
        from matplotlib import cm, colors as mcolors, pyplot as plt

        pos = nx.spring_layout(
            self.recomputable_node_only_graph_with_larger_graph_context,
            k=layout_k,
            iterations=layout_iterations,
        )
        # pos = nx.spectral_layout(graph_with_indirect_edges)
        plt.figure(figsize=(20, 15))

        # Create a dictionary for node labels using the index
        labels = {
            node: self.recomputable_node_only_graph_with_larger_graph_context.nodes[
                node
            ].get("index", node)
            for node in self.recomputable_node_only_graph_with_larger_graph_context.nodes
        }

        # Extract memory values and normalize them
        norm = mcolors.Normalize(
            vmin=min(self.get_knapsack_memory_input()),
            vmax=max(self.get_knapsack_memory_input()),
        )
        cmap = cm.viridis  # type: ignore[attr-defined]

        # Assign colors based on memory
        node_colors = [
            cmap(
                norm(
                    float(
                        self.recomputable_node_only_graph_with_larger_graph_context.nodes[
                            node
                        ]["memory"]
                    )
                )
            )
            for node in self.recomputable_node_only_graph_with_larger_graph_context.nodes
        ]

        # Draw the graph with parsed nodes only
        nx.draw_networkx_nodes(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            node_color=node_colors,
            node_size=300,
            label="Parsed Nodes",
        )
        nx.draw_networkx_edges(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            arrows=True,
            arrowsize=10,
        )
        nx.draw_networkx_labels(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            labels=labels,
            font_size=8,
            font_weight="bold",
        )

        plt.title("Memory Colour Coded Dependency Graph for Recomputable Nodes")
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Memory")
        plt.show()
