# mypy: allow-untyped-defs
# [MODULE DOC] Fuse handlers for FX graph-mode quantization fusion. Defines the abstract
# FuseHandler base, the DefaultFuseHandler that fuses matched modules using the backend's
# fuser method, and a helper mapping fusion patterns to handler classes.
# Exports DefaultFuseHandler and FuseHandler. No import-time side effects.
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fuser_method_mappings import get_fuser_method_new
from torch.ao.quantization.utils import _parent_name, NodePattern, Pattern
from torch.fx.graph import Graph, Node
from torch.nn.utils.parametrize import type_before_parametrizations

from .custom_config import FuseCustomConfig
from .match_utils import MatchAllNode


__all__ = [
    "DefaultFuseHandler",
    "FuseHandler",
]


# ----------------------------
# Fusion Pattern Registrations
# ----------------------------


# Base Pattern Handler
class FuseHandler(ABC):
    # [CLASS DOC] Abstract base class for fusion pattern handlers. A handler is constructed
    # with the matched root Node and must implement fuse(), which emits the fused node
    # into a new graph. Stateless by contract.
    """Base handler class for the fusion patterns"""

    @abstractmethod
    def __init__(self, node: Node):
        # [FUNCTION DOC] Abstract constructor; takes the matched Node. Does nothing here.
        pass

    @abstractmethod
    def fuse(
        self,
        load_arg: Callable,
        named_modules: dict[str, torch.nn.Module],
        fused_graph: Graph,
        root_node: Node,
        extra_inputs: list[Any],
        matched_node_pattern: NodePattern,
        fuse_custom_config: FuseCustomConfig,
        fuser_method_mapping: dict[Pattern, torch.nn.Sequential | Callable],
        is_qat: bool,
    ) -> Node:
        # [FUNCTION DOC] Abstract method: fuse the matched pattern into fused_graph and
        # return the resulting Node. Inputs: load_arg (maps old-graph args to new-graph args),
        # named_modules (name to module mapping, may be mutated), fused_graph, root_node,
        # extra_inputs, matched_node_pattern, fuse_custom_config, fuser_method_mapping, is_qat.
        pass


class DefaultFuseHandler(FuseHandler):
    # [CLASS DOC] Standard fuse handler: replaces the root module of a matched pattern with a
    # fused module produced by the backend's fuser method. Holds no state.
    def __init__(self, node: Node):  # pylint: disable=useless-parent-delegation
        # [FUNCTION DOC] Delegates to the base constructor; the node is unused.
        super().__init__(node)  # type:ignore[safe-super]

    def fuse(
        self,
        load_arg: Callable,
        named_modules: dict[str, torch.nn.Module],
        fused_graph: Graph,
        root_node: Node,
        extra_inputs: list[Any],
        matched_node_pattern: NodePattern,
        fuse_custom_config: FuseCustomConfig,
        fuser_method_mapping: dict[Pattern, torch.nn.Sequential | Callable],
        is_qat: bool,
    ) -> Node:
        # [FUNCTION DOC] Fuse a matched pattern whose root node is a call_module node.
        # Collects the module (or function/method target, or a fresh ReLU module for functional
        # relu, or MatchAllNode otherwise) for each node in the nested pattern, looks up the
        # fuser method by the modules' pre-parametrization types, and builds the fused module
        # with (is_qat, *modules). The fused module replaces the root module on its parent
        # in named_modules (mutation). The root node is copied into fused_graph and any
        # extra_inputs, loaded through load_arg, are appended to its args. Returns the new node.
        # Raises an error if the root node is not a call_module node.
        if root_node.op != "call_module":
            raise AssertionError("Expecting module node to be a call_module Node")
        root_module = named_modules[str(root_node.target)]

        def get_modules(pattern):
            # [FUNCTION DOC] HELPER: convert a nested node pattern into a nested tuple of the
            # corresponding modules/targets (see behavior in the enclosing function).
            """Given a node pattern, extract the corresponding modules
            e.g. input: (relu_node, (bn_node, conv_node))
                 output: (relu_module, (bn_module, conv_module))
            """
            if isinstance(pattern, (tuple, list)):
                n, *args = pattern
                modules: list[torch.nn.Module] = []
                modules.append(get_modules(n))
                modules.extend(get_modules(a) for a in args)
                return tuple(modules)
            else:
                n = pattern
                if n.op == "call_module":
                    return named_modules[n.target]
                elif n.op == "call_function" and n.target is torch.nn.functional.relu:
                    relu = torch.nn.ReLU()
                    relu.training = root_module.training
                    return relu
                elif n.op == "call_function" or n.op == "call_method":
                    return n.target
                else:
                    return MatchAllNode

        # since relu can be used multiple times, we'll need to create a relu module for each match
        matched_modules = get_modules(matched_node_pattern)

        def get_matched_types(m):
            # [FUNCTION DOC] HELPER: replace each module in a nested tuple by its type before
            # parametrization; non-module entries are kept unchanged.
            if isinstance(m, tuple):
                return tuple(map(get_matched_types, m))
            if isinstance(m, torch.nn.Module):
                return type_before_parametrizations(m)
            return m

        matched_module_types = get_matched_types(matched_modules)
        module_parent_name, module_name = _parent_name(root_node.target)
        fuser_method = get_fuser_method_new(matched_module_types, fuser_method_mapping)
        # TODO: change the signature for fuser_method to take matched module patterns
        # as input
        fused_module = fuser_method(is_qat, *matched_modules)
        setattr(named_modules[module_parent_name], module_name, fused_module)
        extra_args = [load_arg(input) for input in extra_inputs]
        node = fused_graph.node_copy(root_node, load_arg)
        args = list(node.args)
        args.extend(extra_args)
        node.args = tuple(args)
        return node


def _get_fusion_pattern_to_fuse_handler_cls(
    backend_config: BackendConfig,
) -> dict[Pattern, Callable]:
    # [FUNCTION DOC] HELPER: build a mapping from each backend-config pattern that defines a
    # fuser method to DefaultFuseHandler. Patterns without a fuser method are omitted.
    fusion_pattern_to_fuse_handlers: dict[Pattern, Callable] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.fuser_method is not None:
            # TODO: is this logic right?
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler
    return fusion_pattern_to_fuse_handlers
