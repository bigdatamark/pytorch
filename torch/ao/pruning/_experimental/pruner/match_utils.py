# [MODULE DOC] FX graph pattern matching helpers for structured pruning: a single-node
# matcher and apply_match, which walks a tuple pattern along node users and returns
# the matched node chain. Re-exports MatchAllNode from quantization utils via import.
# No import-time side effects.
"""
Contains utility functions to check if a pattern is in the graph and return the matching nodes
"""

from typing import Any

import torch
from torch import nn
from torch.ao.quantization.utils import MatchAllNode
from torch.fx import Node
from torch.nn.utils import parametrize


def _match(
    modules: dict[str, nn.ModuleDict],
    node: Node,
    current: nn.Module | Any,
) -> bool:
    # [FUNCTION DOC] HELPER: Returns whether one graph node matches one pattern element.
    # A MatchAllNode subclass always matches; a non-Node never matches otherwise; a
    # Module subclass matches call_module nodes whose module (before parametrizations)
    # has exactly that type; another callable matches call_function nodes with that
    # target; a string matches when equal to the node target.
    r"""
    checks to see if a single node of a pattern matches
    """
    if isinstance(current, type) and issubclass(current, MatchAllNode):
        return True
    if not isinstance(node, Node):
        return False
    if isinstance(current, type) and issubclass(current, torch.nn.Module):
        return (
            node.op == "call_module"
            and parametrize.type_before_parametrizations(modules[node.target])  # type: ignore[index]
            == current
        )
    elif callable(current):
        return node.op == "call_function" and node.target is current
    elif isinstance(current, str):
        return node.target == current
    return False


def apply_match(
    modules: dict[str, nn.ModuleDict],
    pattern: tuple[Any] | Any,
    node: Node,
    matched_node_pattern: list[Node],
) -> list[Node] | None:
    # [FUNCTION DOC] Matches pattern (a tuple of elements or a single element) starting at
    # node. Returns the list of matched nodes (prior matched nodes plus this chain) or
    # None. For a tuple, the first element must match node, then the rest is matched
    # recursively against a user of node.
    # KNOWN ISSUE: only the first user of a node is explored (the loop returns on its
    # first iteration), and the check that rest is None can never be true since rest is a list.
    r"""
    This function will return the matched nodes if the pattern matches the node given
    If there is no match, it will return None
    """
    if isinstance(pattern, tuple):
        if len(pattern) == 1:
            if _match(modules, node, pattern[0]):
                return matched_node_pattern + [node]

        first, *rest = pattern
        if _match(modules, node, first):
            if rest is None:
                return matched_node_pattern + [node]

            for user in node.users:
                return apply_match(
                    modules, tuple(rest), user, matched_node_pattern + [node]
                )
    elif _match(modules, node, pattern):
        return [node]
    return None
