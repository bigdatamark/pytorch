# [MODULE DOC] Matcher that extends SubgraphMatcher so matched nodes can be looked up by name.
# The pattern graph module must return an extra trailing dict of name to pattern node. Exports
# SubgraphMatcherWithNameNodeMap. No import-time side effects.
from torch.fx import Graph, GraphModule, Node
from torch.fx._compatibility import compatibility

from .matcher_utils import InternalMatch, SubgraphMatcher


__all__ = ["SubgraphMatcherWithNameNodeMap"]


def _split_to_graph_and_name_node_map(
    gm: GraphModule,
) -> tuple[GraphModule, dict[str, Node]]:
    # [FUNCTION DOC] HELPER: Strips the trailing name-to-node dict from the output of the pattern graph module.
    # Finds the output node, unflattens with the module output spec, requires a tuple of at least two
    # elements whose last element is a dict (AssertionError otherwise), rewrites the output node to return the
    # remaining flattened values, updates the module pytree output spec, recompiles in place and returns
    # the module with the extracted dict. Mutates the input module.
    from torch.fx.graph import _PyTreeInfo
    from torch.utils._pytree import tree_flatten, tree_unflatten

    name_node_map: dict[str, Node] = {}
    for n in gm.graph.nodes:
        if n.op == "output":
            if gm._out_spec is None:
                raise AssertionError("gm._out_spec is None")
            output = tree_unflatten(n.args[0], gm._out_spec)
            if not isinstance(output, tuple):
                raise AssertionError("Expecting the pattern graph to return a tuple")
            if len(output) < 2:
                raise AssertionError(
                    "Expecting the pattern graph to have at least two outputs"
                )
            *out, name_node_map = output
            flattened, out_spec = tree_flatten(out)
            if not isinstance(name_node_map, dict):
                raise AssertionError(
                    "Expecting the input graph to have a dict output as the last element"
                )
            n.args = (flattened,)
            orig_pytree_info = gm._graph._codegen.pytree_info  # type: ignore[attr-defined]
            gm._graph._codegen.pytree_info = _PyTreeInfo(  # type: ignore[attr-defined]
                orig_pytree_info.orig_args, orig_pytree_info.in_spec, out_spec
            )
    gm.recompile()
    return gm, name_node_map


@compatibility(is_backward_compatible=False)
class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    # [CLASS DOC] SubgraphMatcher variant built from a pattern GraphModule (not Graph) whose extra dict output
    # is removed at construction. State: name_node_map (pattern node names to pattern nodes). After matching,
    # each returned InternalMatch gets its own name_node_map from names to matched target nodes.
    """Extends SubgraphMatcher to support querying the matched subgraph nodes through node name,
    this requires pattern to have specific format (returning and additional dictionary at the output,
    that has node name as key, and the node in the pattern graph as value, see Example for more details)

    Difference with SubgraphMatcher is that it takes a `pattern_gm` GraphModule as input during
    initialization since we need to modify the graph (which requires `recompile` the GraphModule)

    Example::

        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            return relu, {"conv": conv, "relu": relu}


        def target_graph(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu *= 2
            return relu


        pattern_gm = export(pattern, example_inputs).module()
        target_gm = export(target_graph, example_inputs).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        matches = matcher.match(target_gm)
        for match in matches:
            match.name_node_map["conv"].meta["annotation"] = ...

    """

    def __init__(
        self,
        pattern_gm: GraphModule,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
    ) -> None:
    # [FUNCTION DOC] Splits the pattern module into graph and name map (mutating and recompiling the module),
    # stores the map, then initializes the base matcher with the stripped graph and the matching options.
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        self.name_node_map = name_node_map
        super().__init__(
            pattern_gm.graph,
            match_output,
            match_placeholder,
            remove_overlapping_matches,
            ignore_literals,
        )

    def match(self, graph: Graph, node_name_match: str = "") -> list[InternalMatch]:
    # [FUNCTION DOC] Runs the base match on graph (optionally restricted by node_name_match) and fills each
    # returned InternalMatch name_node_map by translating each pattern node in the stored map through the
    # match nodes_map. Returns the list of matches.
        """The returned InternalMatch will have name_node_map populated with a map
        from node name (str) to the target node, e.g.
        ``{"conv": target_conv_node, "relu": target_relu_node}``

        This requires the pattern graph returns an additional
        output of node name to node, e.g. instead of::

            def pattern(...):
                ...
                return relu

        we should do::

            def pattern(...):
                ...
                return relu, {"conv": conv, "relu": relu}
        """
        internal_matches = super().match(graph, node_name_match)
        for internal_match in internal_matches:
            for k, n in self.name_node_map.items():
                internal_match.name_node_map[k] = internal_match.nodes_map[n]
        return internal_matches
