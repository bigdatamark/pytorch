# [MODULE DOC] Small helpers for formatting FX graphs and nodes for logging and error messages.
# Contains a lazy graph-code formatter, a helper that finds the module stack of the first
# function-call node, and a helper that renders the nodes preceding a given node.
# No import-time side effects beyond imports.
import sys
from typing import Any

import torch
from torch._logging import LazyString


def lazy_format_graph_code(
    name: str, gm: torch.fx.GraphModule, maybe_id: int | None = None, **kwargs: Any
) -> LazyString:  # pyrefly: ignore[implicit-any]
    # [FUNCTION DOC] Build a LazyString that, when evaluated, renders the readable code of a
    # GraphModule under a banner made of the name and the optional id. Extra keyword arguments
    # are forwarded to the module's readable-printing method; printing to stdout is turned off
    # by default, and the colored option is forced off when stdout is not a terminal.
    # The kwargs mapping is mutated in place. Nothing is formatted until the string is used.
    """
    Returns a LazyString that formats the graph code.
    """

    def format_name() -> str:
        # [FUNCTION DOC] HELPER: Return the name, followed by the id when an id was given.
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    if "print_output" not in kwargs:
        kwargs["print_output"] = False

    if "colored" in kwargs:
        try:
            if not sys.stdout.isatty():
                kwargs["colored"] = False
        except AttributeError:
            kwargs["colored"] = False

    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(**kwargs),
        )
    )


def _format_graph_code(name: str, filename: str, graph_str: str) -> str:
    # [FUNCTION DOC] HELPER: Join the banner, the source filename and the readable graph text
    # into one string prefixed with a fixed "TRACED GRAPH" header. Pure function.
    """
    Returns a string that formats the graph code.
    """
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> dict[str, Any] | None:
    # [FUNCTION DOC] Scan the graph's nodes in order and return the module-stack metadata of the
    # first function-call node that has one; return None if no such node exists.
    """
    Returns the nn_module_stack of the first call_function node.
    """
    for node in graph.nodes:
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            return node.meta["nn_module_stack"]
    return None


def get_node_context(node: torch.fx.Node, num_nodes: int = 2) -> str:
    # [FUNCTION DOC] Return a newline-joined text of the given node and up to num_nodes - 1
    # nodes before it, oldest first, each rendered with its formatting method. Walking
    # backwards stops early at the root sentinel node.
    # KNOWN ISSUE: the docstring says "last num_nodes nodes in the graph" but the function
    # returns the nodes ending at the given node, not at the end of the graph.
    """
    Returns a string of the last num_nodes nodes in the graph.
    """
    node_contexts = []
    cur = node
    for _ in range(num_nodes):
        # cast to str to handle None return value
        node_contexts.append(str(cur.format_node()))
        if cur.op == "root":
            break
        cur = cur.prev
    return "\n".join(node_contexts[::-1])
