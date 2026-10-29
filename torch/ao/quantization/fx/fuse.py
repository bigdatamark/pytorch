# mypy: allow-untyped-defs
# [MODULE DOC] FX graph-mode fusion pass for quantization. Provides fuse(), which matches
# fusion patterns from a backend config against a GraphModule and rebuilds the graph with
# matched node groups replaced by single fused modules, plus the private match finder.
# Exports fuse and FuseHandler. No import-time side effects.
import warnings
from collections.abc import Callable
from typing import Any

from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
)
from torch.ao.quantization.backend_config.utils import (
    get_fuser_method_mapping,
    get_fusion_pattern_to_extra_inputs_getter,
    get_fusion_pattern_to_root_node_getter,
)
from torch.ao.quantization.utils import NodePattern, Pattern
from torch.fx import GraphModule, map_arg, Node
from torch.fx.graph import Graph

from .custom_config import FuseCustomConfig
from .fuse_handler import _get_fusion_pattern_to_fuse_handler_cls, FuseHandler
from .match_utils import _is_match, MatchAllNode
from .pattern_utils import _sorted_patterns_dict


__all__ = [
    "fuse",
    # TODO: We should make this private in the future
    # This is currently needed for test_public_bindings for some reason
    "FuseHandler",
]


def fuse(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
) -> GraphModule:
    # [FUNCTION DOC] Fuse matched module/function patterns in an FX GraphModule.
    # Args: model (traced GraphModule), is_qat flag, optional FuseCustomConfig and optional
    # BackendConfig (defaults to the native backend config). Dict forms of either config
    # are accepted but emit a FutureWarning and are converted via from_dict.
    # Matches patterns in reverse graph order, then walks the graph building a new Graph:
    # each match's last node is replaced by the fuse handler's output, other matched
    # non-root nodes are dropped, and unmatched nodes (or match-all inputs) are copied.
    # Returns a new GraphModule built from the original and the fused graph. Side effect:
    # the fuse handlers replace modules in the original model via setattr.
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()

    if isinstance(fuse_custom_config, dict):
        warnings.warn(
            "Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported "
            "in a future version. Please pass in a FuseCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)

    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)

    named_modules = dict(model.named_modules())

    if backend_config is None:
        backend_config = get_native_backend_config()

    fusion_pattern_to_fuse_handler_cls = _sorted_patterns_dict(
        _get_fusion_pattern_to_fuse_handler_cls(backend_config)
    )
    fuser_method_mapping = get_fuser_method_mapping(backend_config)
    fusion_pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(
        backend_config
    )
    fusion_pattern_to_extra_inputs_getter = get_fusion_pattern_to_extra_inputs_getter(
        backend_config
    )

    # find fusion
    fusion_pairs = _find_matches(model, model.graph, fusion_pattern_to_fuse_handler_cls)
    # TODO: change this to inplace changes to graph, since we no longer construct
    # new GraphModule anymore
    fused_graph = Graph()
    env: dict[Any, Any] = {}

    def load_arg(a):
        # [FUNCTION DOC] HELPER: map every Node inside argument structure a to its
        # already-created counterpart in the new graph (looked up by node name in env).
        return map_arg(a, lambda node: env[node.name])

    def default_root_node_getter(node_pattern):
        # [FUNCTION DOC] HELPER: default root-node selection. Repeatedly descends into the
        # last element of the nested node pattern until that element is a Node, and returns it.
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]

    for node in model.graph.nodes:
        (
            maybe_last_node,
            pattern,
            matched_node_pattern,
            obj,
            node_to_subpattern,
        ) = fusion_pairs.get(node.name, (None, None, None, None, None))
        # get the corresponding subpattern for the current node
        if node_to_subpattern is not None:
            node_subpattern = node_to_subpattern.get(node, None)
        else:
            node_subpattern = None
        if maybe_last_node is node:
            if obj is None:
                raise AssertionError(
                    "fuse handler object must not be None for matched root node"
                )
            root_node_getter = fusion_pattern_to_root_node_getter.get(
                pattern, default_root_node_getter
            )
            root_node = root_node_getter(matched_node_pattern)  # type: ignore[index]
            extra_inputs_getter = fusion_pattern_to_extra_inputs_getter.get(
                pattern, None
            )
            extra_inputs = []
            if extra_inputs_getter is not None:
                extra_inputs = extra_inputs_getter(matched_node_pattern)
            # TODO: add validation that root_node is a module and has the same type
            # as the root_module in the configuration
            env[node.name] = obj.fuse(
                load_arg,
                named_modules,
                fused_graph,
                root_node,
                extra_inputs,
                matched_node_pattern,  # type: ignore[arg-type]
                fuse_custom_config,
                fuser_method_mapping,
                is_qat,
            )
        elif maybe_last_node is None or node_subpattern is MatchAllNode:
            env[node.name] = fused_graph.node_copy(node, load_arg)
        # node matched in patterns and is not root is removed here

    model = GraphModule(model, fused_graph)
    return model


def _find_matches(
    root: GraphModule,
    graph: Graph,
    pattern_to_fuse_handler_cls: dict[Pattern, Callable],
) -> dict[str, tuple[Node, Pattern, NodePattern, FuseHandler, dict[Node, Any]]]:
    # [FUNCTION DOC] HELPER: find fusion matches in a graph. Iterates nodes in reverse order and
    # tries each pattern (in the given, priority-sorted order) with _is_match; the first match
    # wins and is recorded via apply_match. Returns a mapping from node name to a tuple of
    # (last/root node of match, pattern, matched node pattern, fuse handler instance,
    # mapping from node to the subpattern it matched). Nodes already claimed are skipped.
    modules = dict(root.named_modules())
    # node name -> (root_node, match_value)
    match_map: dict[
        str, tuple[Node, Pattern, NodePattern, FuseHandler, dict[Node, Any]]
    ] = {}
    # a map from node to the matched subpattern
    node_to_subpattern: dict[Node, Any] = {}

    # TODO: dedup with quantization matching function in match_utils.py
    def apply_match(pattern, node, match, matched_node_pattern, node_to_subpattern):
        # [FUNCTION DOC] HELPER: recursively record a pattern match. For a tuple pattern it
        # builds a nested tuple of matched nodes (head node plus argument subpatterns) and
        # appends it to matched_node_pattern. For a leaf pattern, if the node is not yet in
        # match_map, appends the node and (unless the pattern is MatchAllNode) stores its
        # subpattern and a match_map entry. Mutates match_map, node_to_subpattern and
        # matched_node_pattern; earlier matches take precedence.
        if isinstance(pattern, tuple):
            s, *args = pattern
            current_node_pattern: list[Node] = []
            apply_match(s, node, match, current_node_pattern, node_to_subpattern)
            for subpattern, arg in zip(args, node.args):
                apply_match(
                    subpattern, arg, match, current_node_pattern, node_to_subpattern
                )
            matched_node_pattern.append(tuple(current_node_pattern))
        else:
            # the first pattern matches will take precedence
            if node.name not in match_map:
                matched_node_pattern.append(node)
                # MatchAllNode here is actually MatchAllInputNode which should not
                # be added to match_map
                if pattern is not MatchAllNode:
                    node_to_subpattern[node] = pattern
                    root_node, pattern, handler = match
                    match_map[node.name] = (
                        root_node,
                        pattern,
                        matched_node_pattern,
                        handler,
                        node_to_subpattern,
                    )

    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for pattern, fuse_handler_cls in pattern_to_fuse_handler_cls.items():
                matched_node_pattern: list[Node] = []
                if _is_match(modules, node, pattern):
                    apply_match(
                        pattern,
                        node,
                        (node, pattern, fuse_handler_cls(node)),
                        matched_node_pattern,
                        node_to_subpattern,
                    )
                    break

    return match_map
