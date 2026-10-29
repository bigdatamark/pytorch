# [MODULE DOC] Pattern helpers for the FX numeric suite (ns) node matcher.
# Provides the symmetric "related types" relation between ops/modules, the list of
# fusion patterns (stored in reverse order) used to group multi-node sequences, and
# a matcher checking whether a graph node ends such a fusion pattern.
# Defines the type aliases NSFusionElType and NSFusionType. Import-time side effect:
# binds toq to the quantized ops namespace.
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantizeBase, ObserverBase
from torch.ao.quantization.backend_config import get_native_backend_config
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.utils import getattr_from_fqn
from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSNodeTargetType


toq = torch.ops.quantized


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
) -> set[tuple[NSNodeTargetType, NSNodeTargetType]]:
    # [FUNCTION DOC] Build the set of ordered pairs of node target types considered related.
    # Input: mapping from a base name to a set of related ops/modules/functions.
    # Output: set of pairs; every two members of the same set (including a member with
    # itself) appear in both orders. Pure function, no side effects.
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    # TODO(future PR): add the rest of modules and ops here
    type_a_related_to_b: set[tuple[NSNodeTargetType, NSNodeTargetType]] = set()

    for s in base_name_to_sets_of_related_ops.values():
        s_list = list(s)
        # add every bidirectional pair
        for idx_0 in range(len(s_list)):
            for idx_1 in range(idx_0, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))

    return type_a_related_to_b


NSFusionElType = (
    Callable  # call_function or call_module type, example: F.linear or nn.Conv2d
    | str  # call_method name, example: "dequantize"
    | tuple[
        str, Any
    ]  # call_method name and first argument, example: ("to", torch.float16)
)
NSFusionType = (
    tuple[NSFusionElType, NSFusionElType]
    | tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType]
)


def get_reversed_fusions() -> list[tuple[NSFusionType, int]]:
    # [FUNCTION DOC] Return the list of fusion patterns, each paired with a base op index.
    # Patterns come from the native backend config quantization handlers (nested
    # reverse-order patterns are flattened to a flat tuple), multi-op ones are kept, and
    # each is also added with an observer or fake-quant class prepended. Extra hand-written
    # patterns are appended (fp16 emulation linear-relu with base index 1, conv-bn and
    # conv-bn-relu for 1d/2d/3d), each also with observer / fake-quant prefixed variants.
    # The base index is given in non-reversed order. No arguments; builds the result each call.
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
    results: list[tuple[NSFusionType, int]] = []

    # Possible syntaxes:
    # * single op: torch.nn.Conv2d
    # * multiple ops: (torch.nn.ReLU, torch.nn.Conv2d)
    # For fusions, we only care about patterns composed of multiple ops.
    # TODO(future PR): allow customizations from default patterns.
    all_quant_patterns = _get_pattern_to_quantize_handlers(get_native_backend_config())

    default_base_op_idx = 0
    for quant_pattern in all_quant_patterns:
        # TODO: this is a temporary hack to flatten the patterns from quantization so
        # that it works with the ns matcher function, maybe we should use `_is_match`
        # in torch.ao.quantization.fx.match_utils to match the patterns
        if (
            isinstance(quant_pattern, tuple)
            and len(quant_pattern) == 2
            and isinstance(quant_pattern[1], tuple)
            and len(quant_pattern[1]) == 2
        ):
            # flatten the pattern with form (nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))
            quant_pattern = (quant_pattern[0], quant_pattern[1][0], quant_pattern[1][1])

        # Only patterns of multiple ops are fusions, ignore
        # patterns which contain a single op (they get matched
        # without caring about fusions).
        if isinstance(quant_pattern, tuple):
            results.append((quant_pattern, default_base_op_idx))  # type: ignore[arg-type]

        # For each pattern, add additional patterns with observers and
        # fake quants at the end.
        # TODO(future PR): if needed, implement matching for a node
        #   having multiple output observers.
        for cls in (ObserverBase, FakeQuantizeBase):
            if isinstance(quant_pattern, tuple):
                # pyrefly: ignore [not-iterable]
                new_pattern = (cls, *quant_pattern)
            else:
                new_pattern = (cls, quant_pattern)
            results.append((new_pattern, default_base_op_idx))  # type: ignore[arg-type]

    # After this point, results contains values such as
    # [..., ((torch.nn.Relu, torch.nn.Conv2d), 0), ...]

    # Patterns for matching fp16 emulation are not specified in the quantization
    # fusion mappings.  For now, define them here.
    fp16_em_base_op_idx = 1
    patterns_to_add = [
        # linear-relu fp16 emulation:
        # fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        (
            (("to", torch.float16), F.relu, F.linear, "dequantize"),
            fp16_em_base_op_idx,
        ),
        # Conv-BN fusion (this happens outside of quantization patterns,
        # which is why it is defined separately here).
        ((nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
    ]
    for p in patterns_to_add:
        results.append(p)  # type: ignore[arg-type]
        results.append(((ObserverBase, *p[0]), p[1]))  # type: ignore[arg-type]
        results.append(((FakeQuantizeBase, *p[0]), p[1]))  # type: ignore[arg-type]

    return results


def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
    seen_nodes: set[Node],
) -> bool:
    # [FUNCTION DOC] Check whether the pattern ending at end_node matches reversed_fusion.
    # Walks backwards from end_node through first arguments, one fusion element per step.
    # Function elements must equal a call_function target, module elements must be a type
    # the called module is an instance of (module looked up on gm by qualified name),
    # method elements are a name or a (name, second argument) pair compared against the
    # call_method target and args. Returns False on any mismatch, if a node is already in
    # seen_nodes, or if the first argument is not a Node; True if all elements match.
    # Raises an error if a call_module target is not a string. Does not mutate seen_nodes.
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
    cur_node = end_node
    for fusion_idx in range(len(reversed_fusion)):
        # each node can only belong to one matched pattern
        if cur_node in seen_nodes:
            return False

        cur_fusion_el = reversed_fusion[fusion_idx]

        if cur_node.op == "call_function":
            fusion_el_is_fun = (not isinstance(cur_fusion_el, str)) and (
                not isinstance(cur_fusion_el, type)
            )
            if fusion_el_is_fun:
                if cur_node.target != cur_fusion_el:
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == "call_module":
            fusion_el_is_mod = isinstance(cur_fusion_el, type)
            if fusion_el_is_mod:
                if not isinstance(cur_node.target, str):
                    raise AssertionError(f"Expected str, got {type(cur_node.target)}")
                target_mod = getattr_from_fqn(gm, cur_node.target)
                if not isinstance(cur_fusion_el, type):
                    return False
                if not isinstance(target_mod, cur_fusion_el):
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == "call_method":
            fusion_el_is_meth_with_second_arg = (
                isinstance(cur_fusion_el, tuple) and len(cur_fusion_el) == 2
            )
            fusion_el_is_meth_without_args = isinstance(cur_fusion_el, str)
            if fusion_el_is_meth_without_args or fusion_el_is_meth_with_second_arg:
                if fusion_el_is_meth_without_args:
                    if cur_node.target != cur_fusion_el:
                        return False
                else:
                    if not isinstance(cur_fusion_el, tuple):
                        raise AssertionError(
                            f"Expected tuple, got {type(cur_fusion_el)}"
                        )
                    if cur_node.target != cur_fusion_el[0]:
                        return False
                    elif len(cur_node.args) < 2:
                        return False
                    elif cur_node.args[1] != cur_fusion_el[1]:
                        return False

                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False
        else:
            return False

    return True
