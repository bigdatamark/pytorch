# mypy: allow-untyped-defs
# [MODULE DOC] Helper utilities for the FX graph mode quantization flow (prepare/convert).
# Provides node-argument classification (weight/bias, non-tensor args), prepack op lookup,
# unique attribute naming, producer-node extraction, getattr node creation, and custom module
# LSTM/MHA dequant-stub rewriting plus qconfig vs backend dtype-constraint checks.
# Import-time side effect: imports the _decomposed module so quantized_decomposed ops get registered.
import copy
import functools
import operator
import warnings
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigAny, QuantType
from torch.ao.quantization.backend_config import DTypeWithConstraints
from torch.ao.quantization.fake_quantize import (
    FakeQuantizeBase,
    FixedQParamsFakeQuantize,
)
from torch.ao.quantization.observer import (
    _is_activation_post_process,
    FixedQParamsObserver,
    ObserverBase,
)
from torch.ao.quantization.qconfig import (
    float16_dynamic_qconfig,
    float16_static_qconfig,
    qconfig_equals,
)
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
    _assert_and_get_unique_device,
    activation_is_statically_quantized,
)
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node

# importing the lib so that the quantized_decomposed ops are registered
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from .custom_config import PrepareCustomConfig


# TODO: revisit this list. Many helper methods shouldn't be public
__all__ = [
    "all_node_args_except_first",
    "all_node_args_have_no_tensors",
    "assert_and_get_unique_device",
    "collect_producer_nodes",
    "create_getattr_from_value",
    "create_node_from_old_node_preserve_meta",
    "EMPTY_ARG_DICT",
    "get_custom_module_class_keys",
    "get_linear_prepack_op_for_dtype",
    "get_new_attr_name_with_prefix",
    "get_non_observable_arg_indexes_and_types",
    "get_qconv_prepack_op",
    "get_skipped_module_name_and_classes",
    "graph_module_from_producer_nodes",
    "maybe_get_next_module",
    "NodeInfo",
    "node_arg_is_bias",
    "node_arg_is_weight",
    "NON_OBSERVABLE_ARG_DICT",
    "NON_QUANTIZABLE_WEIGHT_OPS",
    "return_arg_list",
    "ObservedGraphModuleAttrs",
]

NON_QUANTIZABLE_WEIGHT_OPS = {
    torch.nn.functional.layer_norm,
    torch.nn.functional.group_norm,
    torch.nn.functional.instance_norm,
}


@dataclass
class ObservedGraphModuleAttrs:
    # [CLASS DOC] Dataclass of metadata attached to an observed (prepared) GraphModule: per-node qconfig and
    # scope maps, prepare custom config, equalization qconfigs, the qconfig mapping, QAT flag, set of
    # observed node names, and standalone-module flags/input-output quantized index lists (default off/None).
    node_name_to_qconfig: dict[str, QConfigAny]
    node_name_to_scope: dict[str, tuple[str, type]]
    prepare_custom_config: PrepareCustomConfig
    equalization_node_name_to_qconfig: dict[str, Any]
    qconfig_mapping: QConfigMapping
    is_qat: bool
    observed_node_names: set[str]
    is_observed_standalone_module: bool = False
    standalone_module_input_quantized_idxs: list[int] | None = None
    standalone_module_output_quantized_idxs: list[int] | None = None


def node_arg_is_weight(node: Node, arg: Any) -> bool:
    # [FUNCTION DOC] Return True if arg is the weight of node: matches args at the weight_index stored in
    # node.meta target_dtype_info (when in range), else the 'weight' keyword argument. Identity comparison.
    """Returns if node arg is weight"""
    weight_index = None
    if "target_dtype_info" in node.meta:
        weight_index = node.meta["target_dtype_info"].get("weight_index", None)
    if (
        weight_index is not None
        and weight_index < len(node.args)
        and node.args[weight_index] is arg
    ):
        return True
    return node.kwargs.get("weight") is arg


def node_arg_is_bias(node: Node, arg: Any) -> bool:
    # [FUNCTION DOC] Return True if arg is the bias of node: matches args at the bias_index stored in
    # node.meta target_dtype_info (when in range), else the 'bias' keyword argument. Identity comparison.
    """Returns if node arg is bias"""
    bias_index = None
    if "target_dtype_info" in node.meta:
        bias_index = node.meta["target_dtype_info"].get("bias_index", None)
    if (
        bias_index is not None
        and bias_index < len(node.args)
        and node.args[bias_index] is arg
    ):
        return True
    return node.kwargs.get("bias") is arg


def get_custom_module_class_keys(
    custom_module_mapping: dict[QuantType, dict[type, type]],
) -> list[Any]:
    # [FUNCTION DOC] Collect the unique float custom module classes across the STATIC, DYNAMIC and WEIGHT_ONLY
    # entries of a custom module mapping (missing quant types are ignored). Returns them as a list in
    # unspecified order (set-based dedup).
    r"""Get all the unique custom module keys in the custom config dict.

    Example input::

        {
            QuantType.STATIC: {CustomModule1: ObservedCustomModule},
            QuantType.DYNAMIC: {CustomModule2: DynamicObservedCustomModule},
            QuantType.WEIGHT_ONLY: {CustomModule3: WeightOnlyObservedCustomModule},
        }

    Example output::

        # extract the keys across all inner STATIC, DYNAMIC, and WEIGHT_ONLY dicts
        [CustomModule1, CustomModule2, CustomModule3]
    """
    # using set to dedup
    float_custom_module_classes: set[Any] = set()
    for quant_mode in [QuantType.STATIC, QuantType.DYNAMIC, QuantType.WEIGHT_ONLY]:
        quant_mode_custom_module_config = custom_module_mapping.get(quant_mode, {})
        quant_mode_custom_module_classes = set(quant_mode_custom_module_config.keys())
        float_custom_module_classes |= quant_mode_custom_module_classes
    return list(float_custom_module_classes)


def get_linear_prepack_op_for_dtype(dtype):
    # [FUNCTION DOC] Map a dtype to the quantized linear prepack op: float16 -> linear_prepack_fp16, qint8 -> linear_prepack.
    # Raises a generic Exception for any other dtype.
    if dtype == torch.float16:
        return torch.ops.quantized.linear_prepack_fp16
    elif dtype == torch.qint8:
        return torch.ops.quantized.linear_prepack
    else:
        raise Exception("can't get linear prepack op for dtype:", dtype)  # noqa: TRY002


def get_qconv_prepack_op(conv_op: Callable) -> Callable:
    # [FUNCTION DOC] Map a functional conv/conv_transpose op (1d/2d/3d) to its quantized prepack op.
    # Raises AssertionError if the op is not in the table.
    prepack_ops = {
        torch.nn.functional.conv1d: torch.ops.quantized.conv1d_prepack,
        torch.nn.functional.conv2d: torch.ops.quantized.conv2d_prepack,
        torch.nn.functional.conv3d: torch.ops.quantized.conv3d_prepack,
        torch.nn.functional.conv_transpose1d: torch.ops.quantized.conv_transpose1d_prepack,
        torch.nn.functional.conv_transpose2d: torch.ops.quantized.conv_transpose2d_prepack,
        torch.nn.functional.conv_transpose3d: torch.ops.quantized.conv_transpose3d_prepack,
    }
    prepack_op = prepack_ops.get(conv_op)
    if prepack_op is None:
        raise AssertionError(f"Didn't find prepack op for {conv_op}")
    return prepack_op


# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    # [FUNCTION DOC] Return a function that, given a module, yields an unused attribute name made of the prefix
    # (dots replaced with underscores) plus the smallest non-negative integer suffix not already an attribute.
    prefix = prefix.replace(".", "_")

    def get_new_attr_name(module: torch.nn.Module):
        # [FUNCTION DOC] HELPER: Inner closure: probes numeric suffixes starting at zero until the name is not an attribute of module.
        def get_attr_name(i: int):
            # [FUNCTION DOC] HELPER: Builds a candidate attribute name from the prefix and integer i.
            return prefix + str(i)

        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name

    return get_new_attr_name


def collect_producer_nodes(node: Node) -> list[Node] | None:
    # [FUNCTION DOC] Trace back from node through args/kwargs collecting producer nodes, stopping expansion at
    # getattr call_function nodes. Returns the node list, or None if a placeholder (graph input) is reached,
    # since the chain is then not self contained.
    r"""Starting from a target node, trace back until we hit input or
    getattr node. This is used to extract the chain of operators
    starting from getattr to the target node, for example::

        def forward(self, x):
            observed = self.observer(self.weight)
            return F.linear(x, observed)

    collect_producer_nodes(observed) will either return a list of nodes that
    produces the observed node or None if we can't extract a self contained
    graph without free variables(inputs of the forward function).
    """
    nodes = [node]
    frontier = [node]
    while frontier:
        node = frontier.pop()
        all_args = list(node.args) + list(node.kwargs.values())
        for arg in all_args:
            if not isinstance(arg, Node):
                continue
            if arg.op == "placeholder":
                # hit input, can't fold in this case
                return None
            nodes.append(arg)
            if not (arg.op == "call_function" and arg.target is getattr):
                frontier.append(arg)
    return nodes


def graph_module_from_producer_nodes(
    root: GraphModule, producer_nodes: list[Node]
) -> GraphModule:
    # [FUNCTION DOC] Build a new GraphModule rooted at root from a non-empty producer node list. Reverses the list in
    # place (mutation), copies each node into a fresh Graph, and outputs the last copied node.
    # Raises AssertionError if the list is empty.
    r"""Construct a graph module from extracted producer nodes
    from `collect_producer_nodes` function
    Args:
      root: the root module for the original graph
      producer_nodes: a list of nodes we use to construct the graph
    Return:
      A graph module constructed from the producer nodes
    """
    if len(producer_nodes) == 0:
        raise AssertionError("list of producer nodes can not be empty")
    # since we traced back from node to getattr
    producer_nodes.reverse()
    graph = Graph()
    env: dict[Any, Any] = {}

    def load_arg(a):
        # [FUNCTION DOC] HELPER: Remaps node arguments to their copies in the new graph via the env mapping.
        return map_arg(a, lambda node: env[node])

    for producer_node in producer_nodes:
        env[producer_node] = graph.node_copy(producer_node, load_arg)
    graph.output(load_arg(producer_nodes[-1]))
    graph_module = GraphModule(root, graph)
    return graph_module


# TODO: delete
@functools.cache
def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    # [FUNCTION DOC] Cached (functools.cache) wrapper returning the unique device of a module's params/buffers, or None
    # if none; errors when multiple devices are found. Marked TODO delete.
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    return _assert_and_get_unique_device(module)


def create_getattr_from_value(
    module: torch.nn.Module,
    graph: Graph,
    prefix: str,
    value: Any,
    device: torch.device | None = None,
) -> Node:
    # [FUNCTION DOC] Register value (tensor cloned/detached, otherwise converted to a tensor on device) as a new buffer on
    # module under a fresh name derived from prefix, and create a get_attr node for it in graph.
    # If device is None it is inferred from the module. Mutates module; returns the new node.
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)
    attr_name = get_new_attr_name(module)
    if device is None:
        device = assert_and_get_unique_device(module)
    new_value = (
        value.detach().clone()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, device=device)
    )
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


def all_node_args_have_no_tensors(
    node: Node, modules: dict[str, torch.nn.Module], cache: dict[Node, bool]
) -> bool:
    # [FUNCTION DOC] Conservatively decide whether all args of node are non-tensors (primitives); results are memoized
    # in cache. Returns True only when certain; unknown or tensor cases give False. Recurses through
    # observer modules, getitem, ndim/shape getattr and size calls, and list/Node args.
    # Raises AssertionError if a call_module node target is not a string.
    # KNOWN ISSUE: a call_module node whose module is not an activation post process leaves result False,
    # and the second call_module branch is unreachable dead code. Also the non-Node-arg case marks tensor found.
    """
    If we know for sure that all of this node's args have no
    tensors (are primitives), return True.  If we either
    find a tensor or are not sure, return False. Note: this
    function is not exact.
    """
    if cache and node in cache:
        return cache[node]

    result = False  # will be overwritten
    if not isinstance(node, Node):
        result = True
    elif node.op == "placeholder":
        result = False
    elif node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError("node.target must be a string for call_module nodes")
        if _is_activation_post_process(modules[node.target]):
            result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    elif node.op == "call_module":
        result = False
    elif node.op == "call_function" and node.target is operator.getitem:
        result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    elif node.op == "get_attr":
        result = False
    elif node.target is getattr and node.args[1] in ["ndim", "shape"]:
        # x1 = x0.ndim
        result = True
    elif node.op == "call_method" and node.target == "size":
        # x1 = x0.size(0)
        result = True
    else:
        found_one_tensor = False
        for arg in node.args:
            if isinstance(arg, list):
                for list_el in arg:
                    if isinstance(list_el, Node):
                        this_list_el_args_have_no_tensors = (
                            all_node_args_have_no_tensors(list_el, modules, cache)
                        )
                        found_one_tensor = found_one_tensor or (
                            not this_list_el_args_have_no_tensors
                        )
                        # If found_one_tensor is True, there is no point in
                        # recursing further as the end result will always
                        # be True.
                        # TODO(future PR): remove this entire function  and
                        # change to dtype inference without recursion.
                        if found_one_tensor:
                            result = not found_one_tensor
                            if cache:
                                cache[node] = result
                            return result
            elif isinstance(arg, int):
                pass
            else:
                if isinstance(arg, Node):
                    this_arg_args_have_no_tensors = all_node_args_have_no_tensors(
                        arg, modules, cache
                    )
                    found_one_tensor = found_one_tensor or (
                        not this_arg_args_have_no_tensors
                    )
                    # If found_one_tensor is True, there is no point in
                    # recursing further as the end result will always
                    # be True.
                    # TODO(future PR): remove this entire function  and
                    # change to dtype inference without recursion.
                    if found_one_tensor:
                        result = not found_one_tensor
                        if cache:
                            cache[node] = result
                        return result
                else:
                    found_one_tensor = True
            result = not found_one_tensor
    if cache:
        cache[node] = result
    return result


def all_node_args_except_first(node: Node) -> list[int]:
    # [FUNCTION DOC] Return the list of arg indices from 1 up to the number of positional args of node.
    """
    Returns all node arg indices after first
    """
    return list(range(1, len(node.args)))


def return_arg_list(arg_indices: list[int]) -> Callable[[Node], list[int]]:
    # [FUNCTION DOC] Return a function that, for a node, gives those of the given arg_indices that are valid positions in node.args.
    """
    Constructs a function that takes a node as arg and returns the arg_indices
    that are valid for node.args
    """

    def arg_indices_func(node: Node) -> list[int]:
        # [FUNCTION DOC] HELPER: Inner closure filtering arg_indices to those smaller than the node's positional arg count.
        return [i for i in arg_indices if i < len(node.args)]

    return arg_indices_func


NodeInfo = namedtuple("NodeInfo", "op target")

# this dict identifies which indices of a node are non tensors
# so that they can be propagated correctly since inserting observers
# for them would cause errors

NON_OBSERVABLE_ARG_DICT: dict[
    NodeInfo, dict[type | torch.dtype, Callable[[Node], list[int]]]
] = {
    NodeInfo("call_method", "masked_fill"): {
        torch.bool: return_arg_list([1]),
        float: return_arg_list([2]),
    },
    NodeInfo("call_method", "permute"): {int: all_node_args_except_first},
    NodeInfo("call_method", "repeat"): {int: all_node_args_except_first},
    NodeInfo("call_method", "reshape"): {int: all_node_args_except_first},
    NodeInfo("call_method", "size"): {int: return_arg_list([1])},
    NodeInfo("call_method", "transpose"): {int: all_node_args_except_first},
    NodeInfo("call_method", torch.transpose): {int: all_node_args_except_first},
    NodeInfo("call_method", "unsqueeze"): {int: return_arg_list([1])},
    NodeInfo("call_method", "unsqueeze_"): {int: return_arg_list([1])},
    NodeInfo("call_method", torch.unsqueeze): {int: return_arg_list([1])},
    NodeInfo("call_method", "view"): {int: all_node_args_except_first},
}

EMPTY_ARG_DICT: dict[type | torch.dtype, Callable[[Node], list[int]]] = {}


def get_non_observable_arg_indexes_and_types(
    node: Node,
) -> dict[type | torch.dtype, Callable[[Node], list[int]]]:
    # [FUNCTION DOC] Look up the node (op, target) in NON_OBSERVABLE_ARG_DICT and return its mapping from non-float
    # type/dtype to an index-selecting function; returns the empty mapping when not listed.
    """
    Returns a dict with of non float tensor types as keys and values which correspond to a
    function to retrieve the list (which takes the node as an argument)
    """
    info = NodeInfo(node.op, node.target)

    return NON_OBSERVABLE_ARG_DICT.get(info, EMPTY_ARG_DICT)


def maybe_get_next_module(
    node: Node,
    modules: dict[str, nn.Module],
    target_module_type: type[nn.Module] | None = None,
    target_functional_type: Any = None,
) -> Node | None:
    # [FUNCTION DOC] Return the first user of node that is a call_module whose module is an instance of
    # target_module_type, or a call_function whose target equals target_functional_type; else None.
    # Checks are skipped for a type that is None.
    """Gets the next module that matches what is needed in
    is_target_module_type if it exists

    Args:
        node: The node whose users we want to look at
        target_module_type: Module type that we want to check
        target_functional_type: Functional type that we want to check
    """

    for user in node.users:
        if (
            user.op == "call_module"
            and target_module_type is not None
            and isinstance(modules[str(user.target)], target_module_type)
        ):
            return user
        elif (
            user.op == "call_function"
            and target_functional_type is not None
            and user.target == target_functional_type
        ):
            return user

    return None


def create_node_from_old_node_preserve_meta(
    quantized_graph: Graph,
    create_node_args: tuple[Any, ...],
    old_node: Node,
) -> Node:
    # [FUNCTION DOC] Create a node in quantized_graph from create_node_args and copy stack_trace from old_node. Returns the new node.
    """
    Creates `new_node` and copies the necessary metadata to it from `old_node`.
    """
    new_node = quantized_graph.create_node(*create_node_args)
    new_node.stack_trace = old_node.stack_trace
    return new_node


def get_skipped_module_name_and_classes(
    prepare_custom_config: PrepareCustomConfig, is_standalone_module: bool
) -> tuple[list[str], list[type[Any]]]:
    # [FUNCTION DOC] Return (names, classes) of modules to skip tracing: copies of the non-traceable names/classes from
    # the prepare custom config; for non-standalone modules also adds standalone module names/classes
    # and the float custom module classes.
    skipped_module_names = copy.copy(prepare_custom_config.non_traceable_module_names)
    skipped_module_classes = copy.copy(
        prepare_custom_config.non_traceable_module_classes
    )
    if not is_standalone_module:
        # standalone module and custom module config are applied in top level module
        skipped_module_names += list(
            prepare_custom_config.standalone_module_names.keys()
        )
        skipped_module_classes += list(
            prepare_custom_config.standalone_module_classes.keys()
        )
        skipped_module_classes += get_custom_module_class_keys(
            prepare_custom_config.float_to_observed_mapping
        )

    return skipped_module_names, skipped_module_classes


def _is_custom_module_lstm(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    qconfig: QConfigAny = None,
    # QuantizeHandler, but we cannot include the type here due to circular imports
    qhandler: Any | None = None,
) -> bool:
    # [FUNCTION DOC] HELPER: True if node is a custom-module LSTM. With a qconfig and a QuantizeHandler: torch.nn.LSTM, statically
    # quantized activation and handler.is_custom_module(); otherwise: quantizable LSTM instance.
    # Raises AssertionError if qhandler is not a QuantizeHandler.
    """
    Return whether this refers to the custom module LSTM flow.
    """
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        if not isinstance(
            qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler
        ):  # type: ignore[attr-defined]
            raise AssertionError("qhandler must be a QuantizeHandler when provided")
        return (
            isinstance(mod, torch.nn.LSTM)
            and activation_is_statically_quantized(qconfig)
            and qhandler.is_custom_module()
        )
    else:
        return isinstance(mod, torch.ao.nn.quantizable.LSTM)


def _is_custom_module_mha(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    qconfig: QConfigAny = None,
    # QuantizeHandler, but we cannot include the type here due to circular imports
    qhandler: Any | None = None,
) -> bool:
    # [FUNCTION DOC] HELPER: Same as the LSTM check but for MultiheadAttention (nn.MultiheadAttention with static qconfig and
    # custom-module handler, or quantizable MultiheadAttention). Raises AssertionError on a bad qhandler.
    """
    Return whether this refers to the custom module MultiheadAttention flow.
    """
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        if not isinstance(
            qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler
        ):  # type: ignore[attr-defined]
            raise AssertionError("qhandler must be a QuantizeHandler when provided")
        return (
            isinstance(mod, torch.nn.MultiheadAttention)
            and activation_is_statically_quantized(qconfig)
            and qhandler.is_custom_module()
        )
    else:
        return isinstance(mod, torch.ao.nn.quantizable.MultiheadAttention)


def _get_module(
    node: Node, named_modules: dict[str, torch.nn.Module]
) -> torch.nn.Module | None:
    # [FUNCTION DOC] HELPER: Return the module for a call_module node found in named_modules, else None.
    """
    If `node` refers to a call_module node, return the module, else None.
    """
    if node.op == "call_module" and str(node.target) in named_modules:
        return named_modules[str(node.target)]
    else:
        return None


def _insert_dequant_stub(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    # [FUNCTION DOC] HELPER: Attach a new DeQuantStub (unique 'dequant_stub_' name) to model and named_modules, and insert a
    # call_module node for it right after node in graph. Returns the new node.
    """
    Attach a `DeQuantStub` to the model and create a node that calls this
    `DeQuantStub` on the output of `node`, similar to how observers are inserted.
    """
    prefix = "dequant_stub_"
    get_new_dequant_stub_name = get_new_attr_name_with_prefix(prefix)
    dequant_stub_name = get_new_dequant_stub_name(model)
    dequant_stub = DeQuantStub()
    setattr(model, dequant_stub_name, dequant_stub)
    named_modules[dequant_stub_name] = dequant_stub
    with graph.inserting_after(node):
        return graph.call_module(dequant_stub_name, (node,))


def _insert_dequant_stubs_for_custom_module_lstm_output(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    # [FUNCTION DOC] HELPER: Rewrite graph so the nested tuple output of a custom LSTM node is split via getitem into output and
    # (hidden0, hidden1), each dequantized by a DeQuantStub, and recombined by tuple nodes. Reroutes other
    # users of the LSTM node to the recombined tuple, then simplifies via tuple/getitem rerouting.
    # Returns the recombined top-level tuple node. Mutates graph and model.
    """
    Insert DeQuantStubs after each internal output node of custom module LSTM.

    Custom module LSTM outputs are nested tuples of the structure (output, (hidden0, hidden1)),
    Since we cannot dequantize a tuple as a whole, we must first break down the tuple into its
    components through `getitem`. This function transforms the graph as follows:

      (1) Split the LSTM node into (output, (hidden0, hidden1))
      (2) Insert a DeQuantStub after each internal node
      (3) Recombine the DeQuantStubs into the same structure as before
      (4) Reroute all consumers of the original LSTM node and its sub-nodes
          (e.g. lstm[0])

    Before:
                   lstm_output
                        |
                        v
                  original_user(s)
    After:
                   lstm_output
                  /           \\
                 /  (getitem)  \\
                /               \\
               v                 v
             output            hidden
               |               /   \\
         (DeQuantStub)        (getitem)
               |             /       \\
               v            v         v
           output_dq     hidden0    hidden1
               |            |         |
               |    (DeQuantStub) (DeQuantStub)
               |            |         |
               |            v         v
               |      hidden0_dq  hidden1_dq
               |            \\       /
               |              (tuple)
               |              \\   /
               |               v  v
               |             hidden_dq
               \\               /
                \\   (tuple)   /
                 v            v
                 lstm_output_dq
                       |
                       v
                original_user(s)

    For step (4), reroute all users of the original LSTM node(s) as follows:
      lstm_output -> lstm_output_dq
      lstm_output[0] -> output_dq
      lstm_output[1] -> hidden_dq
      lstm_output[1][0] -> hidden0_dq
      lstm_output[1][1] -> hidden1_dq

    Return the node `lstm_output_dq`.
    """
    # (1) Split the LSTM node into (output, (hidden0, hidden1))
    # (2) Insert a DeQuantStub after each internal node
    with graph.inserting_after(node):
        output = graph.call_function(operator.getitem, (node, 0))
        output_dq = _insert_dequant_stub(output, model, named_modules, graph)
    with graph.inserting_after(output_dq):
        hidden = graph.call_function(operator.getitem, (node, 1))
    with graph.inserting_after(hidden):
        hidden0 = graph.call_function(operator.getitem, (hidden, 0))
        hidden0_dq = _insert_dequant_stub(hidden0, model, named_modules, graph)
    with graph.inserting_after(hidden0_dq):
        hidden1 = graph.call_function(operator.getitem, (hidden, 1))
        hidden1_dq = _insert_dequant_stub(hidden1, model, named_modules, graph)

    # (3) Recombine the DeQuantStubs into the same structure as before
    with graph.inserting_after(hidden1_dq):
        hidden_dq = graph.call_function(tuple, ([hidden0_dq, hidden1_dq],))
    with graph.inserting_after(hidden_dq):
        lstm_output_dq = graph.call_function(tuple, ([output_dq, hidden_dq],))

    # (4) Reroute all consumers of the original LSTM node and its sub-nodes
    for user in list(node.users.keys()):
        if user != output and user != hidden:
            user.replace_input_with(node, lstm_output_dq)
    # The getitem and tuple nodes we added here may interfere with reference quantized
    # pattern matching, so we need to redirect the consumers of internal nodes to the
    # corresponding nodes with DeQuantStubs (e.g. lstm_output_dq[0] -> output_dq) attached,
    # in order to preserve reference patterns like "dequantize - consumer - quantize".
    _reroute_tuple_getitem_pattern(graph)
    return lstm_output_dq


def _maybe_get_custom_module_lstm_from_node_arg(
    arg: Node,
    named_modules: dict[str, torch.nn.Module],
) -> Node | None:
    # [FUNCTION DOC] HELPER: If arg reaches a custom module LSTM through one of four dequant/getitem/tuple chains, return the LSTM
    # node; else None. Used to skip input observers for consumers of LSTM outputs.
    """
    Given an argument of a node, if the argument refers to the path through which the node
    is a consumer of custom module LSTM, return the custom module LSTM node, or None otherwise.

    This is used to determine whether a node is a consumer of custom module LSTM, and, if so,
    skip inserting input observers for this node. This is because custom module LSTM produces
    quantized outputs, so inserting an input observer for the consumer of custom module LSTM
    would unnecessarily quantize the outputs again.

      lstm -> consumer

    In practice, however, custom module LSTM outputs a tuple (output, (hidden0, hidden1)) with
    DeQuantStubs attached to each internal node (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    This tuple can be consumed in one of four ways:

      lstm -> getitem -> DeQuantStub -> consumer                       # consume lstm[0]
      lstm -> getitem -> getitem -> DeQuantStub -> tuple -> consumer   # consume lstm[1]
      lstm -> getitem -> getitem -> DeQuantStub -> consumer            # consume lstm[1][0] or lstm[1][1]
      lstm -> getitem -> DeQuantStub -> tuple -> consumer              # consume lstm

    Thus, we must match against the above patterns instead of simply checking the parent node
    to determine whether this node is a consumer of a custom module LSTM.
    """

    def match_dq(a):
        # [FUNCTION DOC] HELPER: True if a is a DeQuantStub call_module node.
        return isinstance(_get_module(a, named_modules), DeQuantStub)

    def match_lstm(a):
        # [FUNCTION DOC] HELPER: True if a is a custom module LSTM node.
        return _is_custom_module_lstm(a, named_modules)

    def match_getitem(a):
        # [FUNCTION DOC] HELPER: True if a is an operator.getitem call_function node.
        return a.op == "call_function" and a.target is operator.getitem

    def match_tuple(a):
        # [FUNCTION DOC] HELPER: True if a is a tuple-construction call_function node.
        return a.op == "call_function" and a.target is tuple

    def _match_pattern(match_pattern: list[Callable]) -> Node | None:
        # [FUNCTION DOC] HELPER: Walk upward from arg following the given matcher list, stepping into the first list element for
        # tuple nodes and the first arg otherwise; returns the last node reached or None on mismatch.
        """
        Traverse up the graph and match the args one by one.
        If there is a match, return the last matched node, or None otherwise.
        """
        a = arg
        # pyrefly: ignore [bad-assignment]
        for i, match in enumerate(match_pattern):
            if not match(a):
                return None
            # Match next arg, for tuple the arg is a tuple of a list, e.g. ([dq_1, other_node],)
            if i < len(match_pattern) - 1:
                if match is match_tuple:
                    a = a.args[0][0]  # type: ignore[assignment,index]
                else:
                    a = a.args[0]  # type: ignore[assignment]
        # pyrefly: ignore [bad-return]
        return a

    all_match_patterns = [
        [match_dq, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_getitem, match_lstm],
        [match_dq, match_getitem, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_lstm],
    ]

    for p in all_match_patterns:
        matched_node = _match_pattern(p)
        if matched_node is not None:
            return matched_node
    return None


def _reroute_tuple_getitem_pattern(graph: Graph):
    # [FUNCTION DOC] HELPER: Find N consecutive tuple nodes followed by N reversing getitem nodes and redirect users of the
    # last getitem to the originally packed element, bypassing the pair chain. Mutates graph users;
    # raises AssertionError if a matched pattern has unexpected first/last node kinds.
    """
    Search for patterns where N consecutive `tuple` call_function nodes are followed by
    N consecutive `getitem` call_function nodes that are "reverses" of the `tuple` nodes.
    If we find this pattern, reroute the consumers of the last `getitem` to skip these
    N `tuple` and `getitem` nodes.

    Before:

        a   b     c
        |   \\   /
        \\   tuple
         \\   /
          tuple
            |
        getitem(1)
            |
        getitem(0)
            |
            d

    After:

        b
        |
        d
    """

    def find_patterns(
        node: Node,
        index_stack: list[int],
        current_pattern: list[Node],
        matched_patterns: list[list[Node]],
        seen: set[tuple[Node, tuple[int, ...]]],
    ):
        # [FUNCTION DOC] HELPER: Recursive search from node for tuple/getitem patterns using a stack of expected getitem indices;
        # appends completed patterns to matched_patterns, memoizing visited (node, stack) states in seen.
        # Returns matched_patterns (None when returning early on a seen state).
        """
        Traverse the graph recursively to match for the N-tuple - N-getitem patterns,
        starting at the given node.

        We use a stack to keep track of the expected `getitem` indices, since these are
        reversed from the `tuple` indices. In the above example, the stack after
        (b -> tuple -> tuple) will be [0, 1], which will be popped by getitem(1) first
        and then by getitem(0).

        TODO: traverse upwards from the output and handle the case when tuple is not a
        separate node, e.g. graph.call_function(operator.getitem, args=(a, (b, c)))
        """
        if len(index_stack) == 0 and len(current_pattern) > 0:
            matched_patterns.append(copy.copy(current_pattern))
            current_pattern.clear()

        # Avoid duplicating work
        state = (node, tuple(index_stack))
        if state in seen:
            return
        seen.add(state)

        # Iterate through users of this node to find tuple/getitem nodes to match
        for user in node.users:
            if user.op == "call_function" and user.target is tuple:
                for i, user_arg in enumerate(user.args[0]):  # type: ignore[arg-type]
                    if user_arg == node:
                        index_stack.append(i)
                        current_pattern.append(user)
                        find_patterns(
                            user, index_stack, current_pattern, matched_patterns, seen
                        )
            elif user.op == "call_function" and user.target is operator.getitem:
                if len(index_stack) > 0:
                    if user.args[1] == index_stack[-1]:
                        index_stack.pop()
                        current_pattern.append(user)
                        find_patterns(
                            user, index_stack, current_pattern, matched_patterns, seen
                        )
        return matched_patterns

    # Collect all matched patterns
    matched_patterns: list[list[Node]] = []
    seen: set[tuple[Node, tuple[int, ...]]] = set()  # (node, index_stack)
    for node in graph.nodes:
        find_patterns(node, [], [], matched_patterns, seen)

    # For each pattern, redirect all consumers of the last getitem node to the correct input
    # of the first tuple node
    for pattern in matched_patterns:
        first_tuple = pattern[0]
        last_getitem = pattern[-1]
        if not (first_tuple.op == "call_function" and first_tuple.target is tuple):
            raise AssertionError(
                "first tuple node must be a call_function with target tuple"
            )
        if not (
            last_getitem.op == "call_function"
            and last_getitem.target is operator.getitem
        ):
            raise AssertionError(
                "last getitem node must be a call_function with target operator.getitem"
            )
        last_getitem_index = last_getitem.args[1]
        new_input = first_tuple.args[0][last_getitem_index]  # type: ignore[index]
        for user in list(last_getitem.users.keys()):
            user.replace_input_with(last_getitem, new_input)  # type: ignore[arg-type]


def _get_observer_from_activation_post_process(
    activation_post_process: ObserverBase | FakeQuantizeBase,
) -> ObserverBase:
    # [FUNCTION DOC] HELPER: Return the observer itself, or the inner observer of a FakeQuantizeBase.
    # Raises AssertionError for other types.
    """
    If `activation_post_process` is an observer, return the observer.
    If `activation_post_process` is a fake quantize, return the internal observer.
    """
    if isinstance(activation_post_process, ObserverBase):
        return activation_post_process
    else:
        if not isinstance(activation_post_process, FakeQuantizeBase):
            raise AssertionError(
                "activation_post_process must be an ObserverBase or FakeQuantizeBase"
            )
        return activation_post_process.activation_post_process  # type: ignore[return-value]


def _qconfig_satisfies_dtype_config_constraints(
    qconfig: QConfigAny,
    dtype_with_constraints: DTypeWithConstraints,
    is_activation: bool = True,
) -> bool:
    # [FUNCTION DOC] HELPER: Check a qconfig's activation (or weight) observer against backend DTypeWithConstraints: quant range,
    # minimum scale (eps), and fixed scale/zero point. Instantiates the observer constructor, warns and
    # returns False on violation. Returns True if qconfig is None, backend dtype is None, or dtypes differ.
    # Raises AssertionError if the instantiated object is not an activation post process.
    """
    Return whether `qconfig` satisfies the following constraints from the backend,
    specified through the activation and weight DTypeWithConstraints.

        1. QConfig specified a quantization range that falls within the backend's, if any
        2. QConfig specified a min scale value that is >= the backend's, if any
        3. QConfig specified a FixedQParamsObserver or FixedQParamsFakeQuantize that has
           scale and zero point that match the backend's, if any

    If `is_activation` is True, we check `qconfig.activation`, else we check `qconfig.weight`.
    If `qconfig` or `dtype_with_constraints.dtype` is None, or the dtypes do not match, return True.
    """

    # TODO: log warnings only when the user enabled a debug flag
    def _activation_post_process_satisfies_dtype_config_constraints(
        activation_post_process: ObserverBase | FakeQuantizeBase,
        dtype_with_constraints: DTypeWithConstraints,
        debug_string: str,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Inner check of one observer/fake-quant against the constraints; emits warnings with debug_string and
        # returns False on any violation. fp16 static/dynamic qconfigs are accepted when fixed qparams are required.
        observer = _get_observer_from_activation_post_process(activation_post_process)
        app_quant_min = getattr(observer, "quant_min", None)
        app_quant_max = getattr(observer, "quant_max", None)
        # TODO: for now, just use the existing eps value as scale_min. In the future, we should
        # resolve the differences between the two, either by renaming eps or some other way
        app_scale_min = getattr(observer, "eps", None)
        backend_quant_min = dtype_with_constraints.quant_min_lower_bound
        backend_quant_max = dtype_with_constraints.quant_max_upper_bound
        backend_scale_min = dtype_with_constraints.scale_min_lower_bound
        backend_scale_exact_match = dtype_with_constraints.scale_exact_match
        backend_zero_point_exact_match = dtype_with_constraints.zero_point_exact_match
        # check quantization ranges
        if backend_quant_min is not None and backend_quant_max is not None:
            if app_quant_min is None or app_quant_max is None:
                warnings.warn(
                    f"QConfig {debug_string} must specify 'quant_min' and 'quant_max', ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
            elif app_quant_min < backend_quant_min or app_quant_max > backend_quant_max:
                warnings.warn(
                    f"QConfig {debug_string} quantization range must fall within the backend's:\n"
                    f"QConfig range = ({app_quant_min}, {app_quant_max}), "
                    f"BackendConfig range = ({backend_quant_min}, {backend_quant_max}), "
                    f"ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
        # check scale min
        if backend_scale_min is not None:
            if app_scale_min is None:
                warnings.warn(
                    f"QConfig {debug_string} must specify 'eps', ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
            if app_scale_min < backend_scale_min:
                warnings.warn(
                    f"QConfig {debug_string} eps ({app_scale_min}) must be greater than or equal to "
                    f"the backend's min scale value ({backend_scale_min}), ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
        # check fixed scale and zero point
        if (
            backend_scale_exact_match is not None
            and backend_zero_point_exact_match is not None
        ):
            # For tests only, accept the following qconfigs for now
            # TODO: handle fp16 qconfigs properly
            for accepted_qconfig in [float16_static_qconfig, float16_dynamic_qconfig]:
                if qconfig_equals(qconfig, accepted_qconfig):
                    return True
            suggestion_str = (
                "Please use torch.ao.quantization.get_default_qconfig_mapping or "
                "torch.ao.quantization.get_default_qat_qconfig_mapping. Example:\n"
                '    qconfig_mapping = get_default_qconfig_mapping("fbgemm")\n'
                "    model = prepare_fx(model, qconfig_mapping, example_inputs)"
            )
            if not isinstance(
                activation_post_process, FixedQParamsObserver
            ) and not isinstance(activation_post_process, FixedQParamsFakeQuantize):
                warnings.warn(
                    f"QConfig must specify a FixedQParamsObserver or a FixedQParamsFakeQuantize "
                    f"for fixed qparams ops, ignoring {qconfig}.\n{suggestion_str}",
                    stacklevel=2,
                )
                return False
            if (
                observer.scale != backend_scale_exact_match
                or observer.zero_point != backend_zero_point_exact_match
            ):
                warnings.warn(
                    f"QConfig fixed scale ({observer.scale}) and zero point ({observer.zero_point}) "
                    f"do not match the backend's ({backend_scale_exact_match} and {backend_zero_point_exact_match}), "
                    f"ignoring {qconfig}.\n{suggestion_str}",
                    stacklevel=2,
                )
                return False
        return True

    if qconfig is None or dtype_with_constraints.dtype is None:
        return True

    activation_post_process_ctr = (
        qconfig.activation if is_activation else qconfig.weight
    )
    debug_string = "activation" if is_activation else "weight"
    satisfies_constraints = True
    if activation_post_process_ctr is not None:
        activation_post_process = activation_post_process_ctr()
        if not _is_activation_post_process(activation_post_process):
            raise AssertionError(
                "activation_post_process must be an activation post process"
            )
        # If dtypes don't match, don't check the activation_post_process and return True early
        if activation_post_process.dtype != dtype_with_constraints.dtype:
            return True
        satisfies_constraints = (
            _activation_post_process_satisfies_dtype_config_constraints(
                activation_post_process, dtype_with_constraints, debug_string
            )
        )
    return satisfies_constraints
