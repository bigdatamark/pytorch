# [MODULE DOC] Export pass that rewrites regions governed by a set-grad-enabled call into a call to the
# wrap_with_set_grad_enabled higher-order op with a submodule body. Main entry point:
# replace_set_grad_with_hop_pass; the rest are helpers to detect set-grad nodes, split the graph
# at them, and replace or inline the pieces (shared mechanics in replace_with_hop_pass_util).
# No import-time side effects.
# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

from ..utils import node_inline_, nodes_filter, nodes_first, nodes_map, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_set_grad_enabled_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    # [FUNCTION DOC] HELPER: Predicate that is truthy only when the node is a call_function targeting the
    # low-level set-grad-enabled function. Returns the node itself if it is falsy. No side effects.
    return (
        node
        and node.op == "call_function"
        and node.target is torch._C._set_grad_enabled
    )


def _is_set_grad_enabled_sub_mod(
    node: torch.fx.Node, omit_if_same_with_ambient: bool = False
) -> bool | torch.Tensor:
    # [FUNCTION DOC] HELPER: For a call_module node, checks whether its submodule's first non-placeholder node is a
    # set-grad-enabled call. If so returns True, or, when omit_if_same_with_ambient is set, returns
    # whether the requested grad mode differs from the current global grad mode. Returns False for
    # other nodes. Raises an error if the call_module target is not a string. Reads global grad mode only.
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str target, got {type(node.target)}")
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target is torch._C._set_grad_enabled
        ):
            return (
                first_non_ph.args[0] != torch.is_grad_enabled()  # type: ignore[bad-return]
                if omit_if_same_with_ambient
                else True
            )
    return False


def _replace_with_hop(node: torch.fx.Node) -> None:
    # [FUNCTION DOC] HELPER: Replaces a call_module node with a call to the wrap_with_set_grad_enabled higher-order op.
    # Finds the single set-grad node in the submodule, delegates the rewrite to the shared helper, then
    # erases that node from the submodule graph. Does nothing if the submodule has no set-grad node.
    # Raises assertion errors if there is more than one set-grad node or the node/module is malformed.
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    set_grad_nodes = nodes_filter(sub_graph.nodes, _is_set_grad_enabled_node)
    if len(set_grad_nodes) > 0:
        if len(set_grad_nodes) != 1:
            raise AssertionError(
                f"expected exactly 1 set_grad node, got {len(set_grad_nodes)}"
            )
        set_grad_node = set_grad_nodes[0]
        _replace_with_hop_helper(node, set_grad_node, wrap_with_set_grad_enabled)
        sub_graph.erase_node(set_grad_node)


def _remove_set_grad_and_inline(node: torch.fx.Node) -> None:
    # [FUNCTION DOC] HELPER: Erases every set-grad node from the call_module node's submodule graph, then inlines the
    # submodule's body into the parent graph in place of the call. Used when the grad mode already matches
    # the ambient one. Mutates both graphs. Raises assertion errors on a malformed node.
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    nodes_map(
        sub_graph.nodes,
        lambda n: sub_graph.erase_node(n) if _is_set_grad_enabled_node(n) else n,
    )
    node_inline_(node)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    # [FUNCTION DOC] HELPER: If no set-grad nodes exist returns inputs unchanged. Otherwise splits the graph into sequential
    # submodules at each set-grad node and, per call_module node, either replaces it with the set-grad
    # higher-order op (when the requested mode differs from the ambient grad mode) or strips the set-grad
    # node and inlines it. Returns the new graph module and a signature copy with fixed output names.
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
    need_replacing = any(_is_set_grad_enabled_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # sequential_split returns a new graph module that could have different output
    # args names. We need to fix the graph signature.
    new_gm = sequential_split(gm, _is_set_grad_enabled_node)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: Replaces the submodule call with the higher-order op if its set-grad mode differs
        # from the ambient grad mode, otherwise removes the set-grad node and inlines the submodule.
        if _is_set_grad_enabled_sub_mod(node, omit_if_same_with_ambient=True):
            _replace_with_hop(node)
        else:
            _remove_set_grad_and_inline(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_set_grad_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    # [FUNCTION DOC] Public pass entry point. Takes a graph module and optional export graph signature and returns a
    # (graph module, signature) pair where regions that change grad mode are expressed as
    # wrap_with_set_grad_enabled higher-order op calls. Applies the split-and-replace step to the top level,
    # then recursively to nested graph-module attributes. Mutates and recompiles modules.
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
