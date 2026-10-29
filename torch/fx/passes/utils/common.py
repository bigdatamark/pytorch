# [MODULE DOC] Helpers for lifting a subgraph into a standalone GraphModule and comparing graphs.
# Exports HolderModule, lift_subgraph_as_module and compare_graphs. No import-time side effects.
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.nn import Module


__all__ = ["HolderModule", "lift_subgraph_as_module", "compare_graphs"]


@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    # [CLASS DOC] Module whose only role is to hold submodules/attributes copied from an original module
    # so that dotted attribute paths used by a subgraph resolve. Built from a mapping of name to module
    # (values may be None); each entry is registered as a child module.
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """

    def __init__(self, d: dict[str, Module | None]) -> None:
    # [FUNCTION DOC] Initializes the module and registers every entry of the mapping as a named child module.
        super().__init__()
        for k, v in d.items():
            self.add_module(k, v)


@compatibility(is_backward_compatible=False)
def lift_subgraph_as_module(
    gm: GraphModule,
    subgraph: Graph,
    comp_name: str = "",
    class_name: str = "GraphModule",
) -> tuple[GraphModule, dict[str, str]]:
    # [FUNCTION DOC] Builds a GraphModule for the given subgraph. Walks nodes of op call_module or get_attr,
    # creates nested HolderModules along each dotted target path and copies the leaf attribute from the
    # parent gm. Returns the new GraphModule (named class_name) and a mapping from original target to
    # "comp_name.target". Raises AssertionError if a node target is not a string; fails with
    # AttributeError if the target is missing in gm. Shares (does not clone) leaf objects.
    """
    Create a GraphModule for subgraph, which copies the necessary attributes
    from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (:class:`torch.fx.Graph`): a valid subgraph that contains copied nodes from the
            parent graph

        comp_name (str): name for the new component

        class_name (str): name for the submodule

    """

    # Loop through all module calls (call_module) and param fetches (get_attr)
    # in this component, creating HolderModules as necessary to match the path.
    # e.g. if in the original module there's a get_attr node fetches "conv.weight".
    # We create a HolderModule as root -> add a HolderModule named "conv" ->
    # make "weight" an attribute of "conv" HolderModule and point to conv.weight in
    # the original module.
    submodule = HolderModule({})
    orig_to_split_fqn_mapping: dict[str, str] = {}
    for n in subgraph.nodes:
        if n.op not in ("call_module", "get_attr"):
            continue

        target = n.target
        if not isinstance(target, str):
            raise AssertionError(f"Expected str target, got {type(target)}")
        target_name_parts = target.split(".")
        curr = submodule
        orig_gm = gm

        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                curr.add_module(name, HolderModule({}))

            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)

        leaf_node_name = target_name_parts[-1]
        leaf_node = getattr(orig_gm, leaf_node_name)

        orig_to_split_fqn_mapping[target] = f"{comp_name}.{target}"
        # Relies on custom __setattr__ magic.
        setattr(curr, leaf_node_name, leaf_node)

    return GraphModule(submodule, subgraph, class_name), orig_to_split_fqn_mapping


@compatibility(is_backward_compatible=False)
def compare_graphs(left: Graph, right: Graph) -> bool:
    # [FUNCTION DOC] Returns True if a SubgraphMatcher built from left (matching output and placeholders)
    # finds at least one match in right, i.e. the graphs are structurally identical.
    # KNOWN ISSUE: the docstring claims identical node sets, but matching is subgraph-based, so a larger
    # right graph containing left as a subgraph may also match.
    """
    Return True if two graphs are identical, i.e they
        - have the same number of outputs in the same order
        - have the same number of inputs in the same order
        - have the same set of nodes, and identical connectivity
    """

    matcher = SubgraphMatcher(left, match_output=True, match_placeholder=True)
    matches = matcher.match(right)

    return len(matches) > 0
