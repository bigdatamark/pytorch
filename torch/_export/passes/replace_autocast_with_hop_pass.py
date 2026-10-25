# [MODULE DOC] Export pass that rewrites regions delimited by autocast enter/exit
# calls into a call to the wrap_with_autocast higher-order op whose body is a
# submodule. Main entry point: replace_autocast_with_hop_pass; the rest are helpers
# that detect autocast nodes, split the graph at autocast boundaries and replace or
# inline the resulting submodules. Shared mechanics live in replace_with_hop_pass_util.
# No import-time side effects.
# mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._higher_order_ops.wrap import wrap_with_autocast

from ..utils import node_inline_, nodes_filter, nodes_first, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    # [FUNCTION DOC] HELPER: Predicate that is truthy when the node is a call_function whose
    # target is either the private enter-autocast or exit-autocast function.
    # Returns the (falsy) node itself if node is falsy, otherwise a bool. No side effects.
    return (
        node
        and node.op == "call_function"
        and node.target
        in [
            torch.amp.autocast_mode._enter_autocast,
            torch.amp.autocast_mode._exit_autocast,
        ]
    )


def _is_enter_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    # [FUNCTION DOC] HELPER: Predicate that is truthy only when the node is a call_function
    # targeting the private enter-autocast function. No side effects.
    return (
        node
        and node.op == "call_function"
        and node.target is torch.amp.autocast_mode._enter_autocast
    )


def _is_exit_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    # [FUNCTION DOC] HELPER: Predicate that is truthy only when the node is a call_function
    # targeting the private exit-autocast function. No side effects.
    return (
        node
        and node.op == "call_function"
        and node.target is torch.amp.autocast_mode._exit_autocast
    )


def _is_autocast_sub_mod(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the node is a call_module whose submodule's first
    # non-placeholder node is an enter-autocast call; False for any other node.
    # Raises an error if a call_module target is not a string. Read-only.
    # The existing TODO notes that it does not compare against the current ambient autocast state.
    """
    Check if the first non-placeholder node is `torch.amp.autocast_mode._enter_autocast`.
    """
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
            and first_non_ph.target is torch.amp.autocast_mode._enter_autocast
        ):
            # TODO: check if current auto-cast type is the same as the args of
            # _enter_autocast. If so, return False, i.e. do not create a submodule.
            return True
    return False


def _check_valid_autocast_block(
    enter_autocast_node: torch.fx.Node, exit_autocast_node: torch.fx.Node
) -> None:
    # [FUNCTION DOC] HELPER: Validates that the first argument is an enter-autocast node, the second
    # is an exit-autocast node, and that the exit node's first argument is that enter node.
    # Raises an assertion error on any violation; returns nothing; no side effects.
    if not _is_enter_autocast_node(enter_autocast_node):
        raise AssertionError(
            f"expected enter_autocast node, got {enter_autocast_node.target}"
        )
    if not _is_exit_autocast_node(exit_autocast_node):
        raise AssertionError(
            f"expected exit_autocast node, got {exit_autocast_node.target}"
        )
    if exit_autocast_node.args[0] != enter_autocast_node:
        raise AssertionError(
            "exit_autocast_node.args[0] must match enter_autocast_node"
        )


def _replace_with_hop(node: torch.fx.Node) -> None:
    # [FUNCTION DOC] HELPER: Replaces a call_module node whose submodule holds an autocast region with a
    # call to the wrap_with_autocast higher-order op. Looks up the submodule, collects its autocast
    # nodes (needs at least an enter and an exit; first is enter, last is exit, validated), delegates
    # the rewrite to the shared helper, then erases the enter and exit nodes from the submodule graph.
    # Does nothing if the submodule has no autocast nodes. Mutates the parent graph and submodule graph.
    # Raises assertion errors on malformed input (non call_module node, no owning module, one autocast node).
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
    autocast_nodes = nodes_filter(sub_graph.nodes, _is_autocast_node)
    if len(autocast_nodes) > 0:
        if len(autocast_nodes) <= 1:
            raise AssertionError(
                f"need at least an enter node and an exit node, got {len(autocast_nodes)}"
            )
        enter_autocast_node = autocast_nodes[0]
        exit_autocast_node = autocast_nodes[-1]
        _check_valid_autocast_block(enter_autocast_node, exit_autocast_node)

        _replace_with_hop_helper(node, enter_autocast_node, wrap_with_autocast)
        sub_graph.erase_node(exit_autocast_node)
        sub_graph.erase_node(enter_autocast_node)


def _split_autocast(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] HELPER: Returns a new graph module (input not mutated) split into sequential submodules
    # at outermost autocast boundaries. A callback tracks a stack of open enter nodes; a new
    # submodule starts at an outermost enter node and at the first node after the matching outermost exit.
    # Nested regions stay inside their outer one. Raises assertion errors on an exit without an
    # enter or an exit whose first argument is not the most recent enter. Delegates to sequential_split.
    """
    split_autocast creates a new graph module that splits the input graph module into multiple submodules
    based on the `_enter_autocast` and `_exit_autocast` nodes. It doesn't mutate the input graph module.

    Nodes between the **outer-most** `_enter_autocast` and `_exit_autocast(_enter_autocast)` are split
    into a submodule. Nested autocast regions are not split.
    `_enter_autocast` and `_exit_autocast(_enter_autocast)` nodes are in the submodule as well.

    Below is an example of splitting. A, B, C, D, E are blocks of non-autocast nodes in the original graph
    module. Nodes marked with the same number are grouped into the same submodule.
    A               # 0
    enter_autocast  # 1
    B               # 1
    exit_autocast   # 1
    C               # 2
    enter_autocast  # 3
    D               # 3
    exit_autocast   # 3
    E               # 4
    """
    enter_autocast_node_stack: list[torch.fx.Node] = []
    first_node_after_outer_most_exit: bool = False

    def node_call_back(node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Per-node split callback. Returns True when this node must begin a new
        # submodule (outermost enter node, or first node after an outermost exit), else False.
        # Updates the enclosing enter-node stack and the after-exit flag. Raises assertion errors
        # on unbalanced or mismatched enter/exit nodes.
        nonlocal enter_autocast_node_stack, first_node_after_outer_most_exit
        increment_id = False
        if first_node_after_outer_most_exit or (
            len(enter_autocast_node_stack) == 0 and _is_enter_autocast_node(node)
        ):
            if len(enter_autocast_node_stack) != 0:
                raise AssertionError(
                    f"expected empty stack, got {len(enter_autocast_node_stack)} items"
                )
            first_node_after_outer_most_exit = False
            increment_id = True
        if _is_enter_autocast_node(node):
            enter_autocast_node_stack.append(node)
        elif _is_exit_autocast_node(node):
            if len(enter_autocast_node_stack) == 0:
                raise AssertionError("enter_autocast_node_stack must not be empty")
            last_enter_autocast_node = enter_autocast_node_stack.pop()
            if node.args[0] != last_enter_autocast_node:
                raise AssertionError("exit node args[0] must match last enter node")
            if len(enter_autocast_node_stack) == 0:
                # next node should be in the next submodule since
                # autocast block ends
                first_node_after_outer_most_exit = True
        return increment_id

    return sequential_split(gm, node_call_back)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    # [FUNCTION DOC] HELPER: If the graph has no autocast nodes, returns the inputs unchanged. Otherwise splits
    # the graph at autocast boundaries and, for each resulting call_module node, replaces it with the
    # autocast higher-order op when it is an autocast submodule, or inlines it back into the parent.
    # Returns the new graph module and an updated copy of the signature (output names fixed by the shared helper).
    """
    Helper function for replace_autocast_with_hop_pass().
    Split the graph module into multiple subgraphs based on the autocast nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    Nodes between `_enter_autocast` and `_exit_autocast(_enter_autocast)` are considered
    as a subgraph.
    """
    need_replacing = any(_is_autocast_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # split_autocast returns a new graph module that could have different output
    # args names. We need to fix the graph signature in `_sequential_split_and_maybe_inline_subgraphs_helper`.
    new_gm = _split_autocast(gm)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node) -> None:
        # [FUNCTION DOC] HELPER: For a call_module node, replaces it with the autocast higher-order op if its
        # submodule starts with an enter-autocast call, otherwise inlines the submodule into the parent graph.
        # Raises assertion errors if the node is not a call_module with a string target.
        if _is_autocast_sub_mod(node):
            _replace_with_hop(node)
        else:
            if node.op != "call_module":
                raise AssertionError(f"expected call_module op, got {node.op}")
            if not isinstance(node.target, str):
                raise AssertionError(f"expected str target, got {type(node.target)}")
            node_inline_(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_autocast_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    # [FUNCTION DOC] Public pass entry point. Takes a graph module and optional export graph signature and returns
    # a (graph module, signature) pair in which autocast regions became wrap_with_autocast higher-order op
    # calls with submodule bodies. Applies the split-and-replace step to the top level, then recursively
    # to every nested graph-module attribute (nested ones get no signature). Mutates modules and recompiles.
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
