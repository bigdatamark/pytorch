# [MODULE DOC]
# FX graph passes that inline invoke_subgraph higher order op calls into their
# caller graph: either only subgraphs with a single call site across the module
# tree, or all of them. Provides counting, submodule hoisting and node-inlining
# helpers. No import-time side effects.
import operator
from collections import Counter
from typing import Any, TYPE_CHECKING

import torch
from torch.fx.graph_module import GraphModule


if TYPE_CHECKING:
    from torch.fx.node import Node


def count_subgraph_uses(gm: GraphModule) -> Counter[str]:
    # [FUNCTION DOC] Walks gm and every nested module that is a GraphModule and
    # counts invoke_subgraph call nodes, keyed by the string form of the
    # identifier argument (second argument). Returns the counter; no mutation.
    """Count invoke_subgraph references by identifier string across all
    descendant GraphModules."""
    counts: Counter[str] = Counter()
    for mod in gm.modules():
        if not isinstance(mod, GraphModule):
            continue
        for node in mod.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        ):
            counts[str(node.args[1])] += 1
    return counts


def inline_single_use_invoke_subgraph(gm: GraphModule) -> GraphModule:
    # [FUNCTION DOC] Pass entry point: computes global use counts over the module
    # tree, then recursively inlines invoke_subgraph nodes whose identifier is used
    # exactly once (and that carry no nested region config). Mutates gm and its
    # submodules in place and returns gm.
    """Inline invoke_subgraph HOPs whose subgraph module has exactly one call
    site across the entire module tree.

    When a subgraph has only a single caller, invoke_subgraph adds overhead
    without any deduplication benefit, so we inline it unconditionally.
    Subgraph modules that are shared across sibling modules (e.g. two AC body
    modules referencing the same nested_compile_region) are counted globally
    by module identity and preserved when referenced more than once.
    """
    global_counts = count_subgraph_uses(gm)
    inline_single_use_recursive(gm, global_counts)
    return gm


def inline_single_use_recursive(gm: GraphModule, global_counts: Counter[str]) -> None:
    # [FUNCTION DOC] HELPER: Recurses into every named GraphModule submodule first,
    # then inlines in gm the invoke_subgraph nodes whose identifier count in
    # global_counts is one and whose referenced subgraph module has no
    # nested_region_config in its meta. Mutates gm in place; returns None.
    # KNOWN ISSUE: named_modules yields all descendants, so deeper modules may be
    # processed more than once (once per ancestor); harmless but redundant.
    # Recursively apply to nested subgraph modules first.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_single_use_recursive(mod, global_counts)

    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    if not invoke_nodes:
        return

    single_use_nodes = [
        node
        for node in invoke_nodes
        if global_counts[str(node.args[1])] == 1
        and not getattr(gm, str(node.args[0].target)).meta.get(
            "nested_region_config", None
        )
    ]
    if not single_use_nodes:
        return

    inline_invoke_subgraph_nodes(gm, single_use_nodes)


def hoist_submodules(gm: GraphModule, subgraph: GraphModule) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Registers each direct child module of subgraph on gm,
    # choosing a unique name by appending a numeric suffix on collision. Returns a
    # mapping from old child name to new name. Side effect: adds modules to gm.
    """Move submodules of *subgraph* onto *gm*, returning a name-remap dict.

    For each direct child module of *subgraph* (e.g. ``fwd_body_0``), register
    it on *gm* under a unique name and return a mapping from the old name to the
    new name so that ``get_attr`` targets copied from *subgraph*'s graph can be
    rewritten.
    """
    name_map: dict[str, str] = {}
    existing_names = {name for name, _ in gm.named_children()}
    for child_name, child_mod in list(subgraph.named_children()):
        new_name = child_name
        counter = 0
        while new_name in existing_names:
            new_name = f"{child_name}_{counter}"
            counter += 1
        gm.add_module(new_name, child_mod)
        existing_names.add(new_name)
        name_map[child_name] = new_name
    return name_map


def inline_invoke_subgraph_nodes(gm: GraphModule, invoke_nodes: list["Node"]) -> None:
    # [FUNCTION DOC] HELPER: For each invoke_subgraph node, hoists the subgraph's
    # submodules onto gm, copies the subgraph's non-placeholder, non-output nodes
    # before the call (placeholders bound to the call operands, get_attr targets
    # renamed per the hoist map), rewires getitem users of the call to the
    # corresponding subgraph outputs, then erases the call, its getitem users and
    # the now-unused get_attr node. Mutates gm's graph; returns None.
    # Assumes the subgraph output is a tuple/list indexed only through getitem.
    """Shared helper that inlines a list of invoke_subgraph nodes."""
    for node in invoke_nodes:
        get_attr_node: torch.fx.Node = node.args[0]  # pyrefly: ignore[bad-assignment]
        operands = node.args[2:]

        subgraph: GraphModule = getattr(gm, str(get_attr_node.target))

        # Hoist sub-submodules (e.g. fwd_body_0, bwd_body_0) to gm so that
        # get_attr nodes copied from the subgraph resolve correctly.
        name_map = hoist_submodules(gm, subgraph)

        env: dict[Node, Any] = dict(
            zip(subgraph.graph.find_nodes(op="placeholder"), operands)
        )

        with gm.graph.inserting_before(node):
            for sub_node in subgraph.graph.nodes:
                if sub_node.op in ("placeholder", "output"):
                    continue
                new_node = gm.graph.node_copy(sub_node, lambda n: env[n])
                if sub_node.op == "get_attr" and sub_node.target in name_map:
                    new_node.target = name_map[sub_node.target]
                env[sub_node] = new_node

        output_values = subgraph.graph.output_node().args[0]

        for user in list(node.users):
            if user.op == "call_function" and user.target is operator.getitem:
                idx = user.args[1]
                user.replace_all_uses_with(env[output_values[idx]])  # pyrefly: ignore
                gm.graph.erase_node(user)

        gm.graph.erase_node(node)

        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)


def inline_invoke_subgraph(gm: GraphModule) -> GraphModule:
    # [FUNCTION DOC] Pass entry point that inlines every invoke_subgraph call in gm.
    # First recursively flattens each nested GraphModule, then inlines the calls
    # in gm itself. Mutates in place and returns gm.
    """Inline all invoke_subgraph HOPs, producing a flat FX graph.

    This is useful when downstream compilers (like vllm-compile) don't support
    HOPs or prefer a flat graph, but we still want the Dynamo tracing-time
    benefits of auto-caching (trace once, stamp out cached calls).
    """
    # Recursively inline nested invoke_subgraph calls inside subgraph
    # modules first, so their graphs are flat before we inline them here.
    for name, mod in gm.named_modules():
        if name and isinstance(mod, GraphModule):
            inline_invoke_subgraph(mod)

    invoke_nodes = list(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.invoke_subgraph
        )
    )
    if not invoke_nodes:
        return gm

    inline_invoke_subgraph_nodes(gm, invoke_nodes)
    return gm
