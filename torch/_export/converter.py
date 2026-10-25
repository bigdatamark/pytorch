# [MODULE DOC] TorchScript to torch.export converter. Takes a TorchScript
# module or function (traced or scripted) plus example inputs, builds a jit
# graph, and translates each TorchScript node into FX nodes (control flow
# becomes torch.cond / higher-order ops, attributes become params, buffers or
# constants), producing an ExportedProgram via TS2EPConverter. Main contents:
# graph preparation helpers, the operator-name tables, TS2FXGraphConverter
# (node-by-node conversion with many convert_* handlers) and TS2EPConverter
# (signature building and program assembly). Import-time side effects: only
# defines module-level tables and a logger.
# mypy: allow-untyped-defs
import builtins
import logging
import operator
import typing
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.export._trace
from torch import _C
from torch._export.passes.replace_quantized_ops_with_standard_ops_pass import (
    replace_quantized_ops_with_standard_ops,
)
from torch.export.dynamic_shapes import _tree_map_with_path, Dim
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    ConstantArgument,
    CustomObjArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx import subgraph_rewriter


log = logging.getLogger(__name__)


def _get_param_count_list(method_graph, args_params):
    # [FUNCTION DOC] HELPER: for each graph input paired with its argument/param
    # value, records how many flattened values it spans: the flattened length for
    # a packed-params typed input, else a boolean saying the value is not None.
    # Returns the list (zip truncates to the shorter sequence).
    param_count_list = []
    for input_, arg_params_ in zip(method_graph.inputs(), args_params):
        if "PackedParams" in str(input_.type()):
            in_vars, _ = torch.jit._flatten(arg_params_)
            param_count_list.append(len(in_vars))
        else:
            param_count_list.append(arg_params_ is not None)

    return param_count_list


def _trace_and_get_graph_from_model(model, args):
    # [FUNCTION DOC] HELPER: traces model with args through the jit tracer
    # (non-strict, not forcing out-of-place ops) with the autocast weight cache
    # temporarily disabled, restoring the previous setting afterwards (not
    # restored if tracing raises). Raises RuntimeError if the model's state dict
    # keys changed during tracing. Returns the trace graph and the model output.
    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()

    # Disable Autocast cache because it replaces kernel's weight and bias
    # by (undesired) constants.
    # No perf impact for when there are reused weights since https://github.com/pytorch/pytorch/pull/85665
    prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    trace_graph, torch_out, _inputs_states = torch.jit._get_trace_graph(
        model,
        args,
        strict=False,
        _force_outplace=False,
        _return_inputs_states=True,
    )
    torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)

    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )

    return trace_graph, torch_out


def _create_jit_graph(
    model: torch.nn.Module | torch.jit.ScriptFunction, args: Sequence[Any]
) -> tuple[torch.Graph, list["_C.IValue"], Any, torch.ScriptModule | None]:
    # [FUNCTION DOC] HELPER: builds a jit graph for model. For a ScriptModule:
    # takes the forward graph, substitutes functions, freezes the module
    # preserving parameters, lists its parameters and propagates input shapes
    # over args plus params; returns (graph, params, None, frozen module); raises
    # RuntimeError if forward is not a script method. For a ScriptFunction: same
    # without freezing; returns (graph, empty params, None, None). For an
    # eager module: traces it, lints the graph, names the parameter inputs after
    # the state dict keys and returns (graph, params, traced output, None).
    if isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        flattened_args = tuple(torch.jit._flatten(tuple(args))[0])
        torch_out = None

        if isinstance(model, torch.jit.ScriptModule):
            try:
                graph = model.forward.graph  # type: ignore[attr-defined]
            except AttributeError as e:
                raise RuntimeError("'forward' method must be a script method") from e
            _C._jit_pass_onnx_function_substitution(graph)
            freezed_module = _C._freeze_module(
                typing.cast(_C.ScriptModule, model._c), preserveParameters=True
            )
            module, params = _C._jit_onnx_list_model_parameters(freezed_module)
            method_graph = module._get_method("forward").graph
            args_params = tuple(args) + tuple(params)
            param_count_list = _get_param_count_list(method_graph, args_params)
            in_vars, _ = torch.jit._flatten(args_params)
            graph = _C._propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), param_count_list, False, False
            )
            return graph, params, torch_out, module

        # torch.jit.ScriptFunction
        params = []
        graph = model.graph
        _C._jit_pass_onnx_function_substitution(graph)
        param_count_list = _get_param_count_list(graph, args)
        graph = _C._propagate_and_assign_input_shapes(
            graph, flattened_args, param_count_list, False, False
        )
        return graph, params, torch_out, None

    graph, torch_out = _trace_and_get_graph_from_model(model, args)
    _C._jit_pass_onnx_lint(graph)
    state_dict = torch.jit._unique_state_dict(model)
    params = list(state_dict.values())
    graph_inputs = list(graph.inputs())
    user_input_num = len(graph_inputs) - len(state_dict)
    param_names = list(state_dict.keys())
    for i, inp in enumerate(graph_inputs):
        if i >= user_input_num:
            inp.setDebugName(param_names[i - user_input_num])
    _C._jit_pass_onnx_function_substitution(graph)
    return graph, params, torch_out, None


def list_add(a, b):
    # [FUNCTION DOC] Returns the concatenation (plus) of a and b; used to
    # implement TorchScript list add as a traceable call.
    return a + b


def list_append(container, element):
    # [FUNCTION DOC] Returns a new list made of container followed by element;
    # does not mutate container (functional replacement for in-place append).
    return container + [element]


def execute_subgraph_from_prim_loop(
    subgraph, iter_idx, len_loop_local_arguments, *args, **kwargs
):
    """
    subgraph: GraphModule from sub-block.
    iter_idx: The index of interaction.
    len_loop_local_arguments: The number of loop local arguments in args.
    """
    # [FUNCTION DOC] Runs one iteration of a converted prim::Loop body.
    # The first len_loop_local_arguments of args are loop-carried values; the
    # remaining args are captured outer values. Calls the subgraph with the
    # captured values first, then the iteration index, then the loop-carried
    # values (plus kwargs) and returns its result.

    # Loop local variables. TS graph create those as inputs because their values
    # are updated inside the loop.
    loop_local_args = args[:len_loop_local_arguments]
    # Global variables that are not passed in as inputs to the loop sub-blocks
    # but are directly used. Most of time, their values are not updated, but
    # the only exception is when there are some operations that perform inplace
    # updates.
    global_args = args[len_loop_local_arguments:]
    return subgraph(*global_args, iter_idx, *loop_local_args, **kwargs)


def inplace_optimize_sym_size_div(gm: torch.fx.GraphModule):
    # [FUNCTION DOC] Rewrites, in place on gm, the pattern symbolic size ->
    # scalar tensor -> truncating scalar division -> integer conversion into a
    # single floor division of the symbolic size by the scale. Mutates the graph
    # module; returns None.
    def pattern(im, dim, scale):
        # [FUNCTION DOC] HELPER: the subgraph pattern to match (size, scalar
        # tensor, trunc division, to int).
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        scalar_tensor = torch.ops.aten.scalar_tensor(sym_size_int)
        div_scalar_mode = torch.ops.aten.div.Scalar_mode(
            scalar_tensor, scale, rounding_mode="trunc"
        )
        int_tensor = torch.ops.aten.Int.Tensor(div_scalar_mode)
        return int_tensor

    def replacement(im, dim, scale):
        # [FUNCTION DOC] HELPER: the replacement subgraph; symbolic size of im
        # along dim integer-divided by scale.
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        return sym_size_int // scale

    subgraph_rewriter.replace_pattern(gm, pattern, replacement)


def is_valid_for_codegen(name):
    # [FUNCTION DOC] Returns False if name starts with a digit, else True.
    # Raises RuntimeError for an empty name.
    if len(name) == 0:
        raise RuntimeError("Empty argument name for codegen")
    if name[0].isdigit():
        return False
    return True


def normalize_name(name: str, prefix: str = "rename") -> str:
    # [FUNCTION DOC] Makes name a valid identifier for FX codegen: replaces dots
    # with underscores and, if the result begins with a digit, prepends the
    # prefix and an underscore. Raises RuntimeError on an empty name.
    name = name.replace(".", "_")
    if is_valid_for_codegen(name):
        return name
    return f"{prefix}_{name}"


def ir_name_to_func_name(name: str) -> str:
    """prim::If -> convert_prim_If"""
    # [FUNCTION DOC] Maps a TorchScript kind string with namespace separator to
    # the converter method name: the prefix convert_ followed by the parts
    # joined with underscores.
    name_list = name.split("::")
    return "convert_" + "_".join(name_list)


def get_node_as_placeholder_or_get_attr(fx_graph, name, is_top_level_graph):
    # [FUNCTION DOC] Creates a node named name in fx_graph: a get_attr node for
    # the top-level graph, otherwise a placeholder (lifted input) for sub-blocks.
    # Mutates fx_graph and returns the new node.
    if is_top_level_graph:
        return fx_graph.get_attr(name)
    return fx_graph.placeholder(name)


_TORCH_DTYPE_TO_ENUM = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.complex32: 8,
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.qint8: 12,
    torch.quint8: 13,
    torch.bfloat16: 15,
}

_TORCH_ENUM_TO_DTYPE = {value: key for key, value in _TORCH_DTYPE_TO_ENUM.items()}


def get_dtype_as_int(tensor):
    """
    prim::dtype has the signature "Tensor a) -> int", where it gets the dtype of
    the tensor and returns the integer corresponding to this dtype based on the
    enum in ScalarType.h
    """
    # [FUNCTION DOC] Returns the integer scalar-type code of tensor's dtype using
    # a local table. Raises RuntimeError for dtypes missing from the table.
    # KNOWN ISSUE: the table omits some types (for example quantized 32-bit and
    # float8 types), so those raise.
    dtype = tensor.dtype
    if dtype not in _TORCH_DTYPE_TO_ENUM:
        raise RuntimeError(f"Unsupported dtype {dtype}")
    return _TORCH_DTYPE_TO_ENUM[dtype]


# Those operators will be automatically populated to an instance method
# of TS2FXGraphConverter with name convert_<namespace>_<opname>().
# Please check __init__ for method population implementations.
kind_to_standard_operators: dict[str, Callable[..., Any]] = {
    "prim::max": builtins.max,
    "prim::min": builtins.min,
    "prim::TupleIndex": operator.getitem,
    "aten::__is__": operator.is_,
    "aten::__isnot__": operator.is_not,
    "aten::__not__": operator.not_,
    "aten::__contains__": operator.contains,
    "prim::dtype": get_dtype_as_int,
    "aten::len": len,
    # Mapping from specialized op to its symbolic counterpart.
    # They currently do not have any other overrides.
    "aten::numel": torch.ops.aten.sym_numel,
    "aten::size": torch.ops.aten.sym_size,
    "aten::storage_offset": torch.ops.aten.sym_storage_offset,
    "aten::stride": torch.ops.aten.sym_stride,
}


def get_ir_value_parent_name_and_attr_name(node):
    # [FUNCTION DOC] For a GetAttr node, returns the tuple (output value debug
    # name, input value debug name, attribute name string).
    irv_parent_name, irv_name = node.input().debugName(), node.output().debugName()
    attr_name = node.s("name")
    return irv_name, irv_parent_name, attr_name


def construct_fqn(ir, ref_map, name_map):
    # [FUNCTION DOC] Builds a dotted fully qualified name by walking from ir up
    # the child-to-parent reference map, collecting each step's attribute name
    # from name_map, and joining them from the outermost to the innermost.
    name_list = []
    while ir in ref_map:
        name_list.append(name_map[ir])
        ir = ref_map[ir]
    return ".".join(reversed(name_list))


def get_block_to_lifted_attrs(
    graph: torch._C.Graph,
) -> tuple[dict[torch._C.Block, set[str]], dict[str, str]]:
    """
    Perform two passes to get a mapping of blocks to a set of FQNs of its lifted attributes.
    When a graph has control flow, the graph will be divided into multiple blocks. We want to convert
    each block to a graph which will be passed into torch.cond. A restriction for torch.cond is that model
    parameters/buffers are expected to be lifted as inputs to the subgraphs. Before converting the model,
    we will run this pass which will:
        1. Figure out which params/buffers are used within blocks through tracing the GetAttr calls.
        2. Process the graph bottom up to find the lifted attributes of each block by taking the union
        of the attributes used in the current block, and the lifted attributes of all its child blocks.

    Returns:
        A mapping of blocks to a set of FQNs of its lifted attributes, and a
        mapping of node names to the FQNs of its lifted attributes.
    """
    # [FUNCTION DOC] Two-pass analysis of graph (see docstring). Pass one walks
    # all blocks recording, for every GetAttr node, its parent value and
    # attribute name. Pass two walks bottom-up and, for each non-top-level
    # block, stores the set of leaf attribute FQNs used in it or its children.
    # Returns (block to FQN set mapping, mapping of value name to attribute
    # name). KNOWN ISSUE: the docstring says the second map holds FQNs, but it
    # holds the single attribute name of each GetAttr output, not full paths.

    # A map from a block to its expected to be lifted arguments.
    blocks_to_lifted_attrs: dict[torch._C.Block, set[str]] = {}

    # Reference map stores the input (i.e., src) and output (i.e., dest) IR of a
    # GetAttr node. By traversing this reference map, we can figure out the
    # full IR aliasing pass and figure out the FQN of an attribute.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_parent_map["%2"] = "%1"
    node_to_parent_map: dict[str, str] = {}

    # Used for reconstructing the FQN of an attribute based on the reference map.
    # In nutshell, for each GetAttr call, GetAttr(input IR, attribute name) -> output IR
    # This name map stores which attribute name is called for a src IR --> dest IR action.
    # E.g., %2 = GetAttr(linear)[%1] --> node_to_attr_name["%2"] = "linear"
    node_to_attr_name: dict[str, str] = {}

    def _dfs_get_attr_dependency(entry):
        """
        First DFS path to construct reference map and name map.
        """
        # [FUNCTION DOC] HELPER: recursively visits entry's nodes and nested
        # blocks, filling the parent map and attribute-name map for every
        # GetAttr node (mutates the enclosing dictionaries).
        for node in entry.nodes():
            if node.kind() == "prim::GetAttr":
                (
                    irv_name,
                    irv_parent_name,
                    attr_name,
                ) = get_ir_value_parent_name_and_attr_name(node)
                node_to_parent_map[irv_name] = irv_parent_name
                node_to_attr_name[irv_name] = attr_name
            for block in node.blocks():
                _dfs_get_attr_dependency(block)

    def _map_blocks_to_lifted_attrs(entry):
        """
        Walk the graph in a bottom-up fashion to build the expected to be
        lifted arguments for each block.
        """
        # [FUNCTION DOC] HELPER: recursively returns the set of leaf attribute
        # FQNs used in entry and all its child blocks; GetAttr outputs that are
        # the parent of another GetAttr are skipped. Records the set for every
        # block except the top-level graph in the enclosing mapping.
        arguments: set[str] = set()
        for node in entry.nodes():
            for block in node.blocks():
                # Recursively build.
                arguments = arguments.union(_map_blocks_to_lifted_attrs(block))
            if node.kind() == "prim::GetAttr":
                irv_name = node.output().debugName()
                # Skip for intermediate GetAttr, which will anyway not result a FQN.
                # E.g., node_to_parent_name: {"%3": "%2", "%2": "%1"}
                #       node_to_attr_name: {"%3": "weight", "%2": "linear", "%1": "self"}
                #       There is only one FQN %3-->%2-->%1: self.linear.weight
                #       %2-->%1 is not a FQN: self.linear
                if irv_name not in set(node_to_parent_map.values()):
                    arguments.add(
                        construct_fqn(irv_name, node_to_parent_map, node_to_attr_name)
                    )
        if not isinstance(entry, torch._C.Graph):  # Skip the top level.
            blocks_to_lifted_attrs[entry] = arguments
        return arguments

    _dfs_get_attr_dependency(graph)
    _map_blocks_to_lifted_attrs(graph)

    return blocks_to_lifted_attrs, node_to_attr_name


def get_attribute_fqn_from_ts_node(
    name_to_attribute_fqn: dict[str, str], node: torch._C.Node
) -> str:
    # [FUNCTION DOC] Returns the attribute FQN for a SetAttr or GetAttr node:
    # the FQN already known for the node's object input (looked up in
    # name_to_attribute_fqn), joined with the node's attribute name by a dot, or
    # just the attribute name if the root FQN is empty. Raises ValueError if the
    # input is unknown and RuntimeError for other node kinds.
    def get_attr(name: str):
        # [FUNCTION DOC] HELPER: looks up name in the FQN mapping; raises
        # ValueError if absent.
        if name in name_to_attribute_fqn:
            return name_to_attribute_fqn[name]
        else:
            raise ValueError(f"Attribute {name} not found")

    if node.kind() == "prim::SetAttr":
        input_name = next(node.inputs()).debugName()
    elif node.kind() == "prim::GetAttr":
        input_name = node.input().debugName()
    else:
        raise RuntimeError(
            f"Unexpected node kind when getting attribute fqn. node: {node} "
        )

    attr_name = node.s("name")
    root_attr_name = get_attr(input_name)
    attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name

    return attr_fqn


def get_op_overload(node: torch._C.Node):
    # [FUNCTION DOC] Resolves the torch.ops overload for a TorchScript node from
    # its schema string: parses namespace, op name and overload name, and
    # returns the matching overload (default overload if the name is empty).
    # Raises AssertionError if the node has no schema and RuntimeError if the
    # operator cannot be found.
    schema_str = node.schema()
    if schema_str == "(no schema)":
        raise AssertionError(f"got empty schema for {node}")
    schema: torch._C.FunctionSchema = torch._C.parse_schema(schema_str)
    ns, op_name = str(schema.name).split("::")
    override = schema.overload_name

    try:
        op_overload_mod = getattr(torch.ops, ns)
        op_overload_packet = getattr(op_overload_mod, op_name)
        if override:
            op_overload = getattr(op_overload_packet, override)
        else:
            op_overload = op_overload_packet.default
    except Exception as e:
        raise RuntimeError(
            f"Unable to find operator {node.kind()} with schema {node.schema()}"
        ) from e

    return op_overload


class TS2FXGraphConverter:
    # [CLASS DOC] Converts one TorchScript graph or block into an FX graph.
    # State: the source graph, shared parameter/buffer/constant/attribute-FQN
    # tables, the FX graph under construction, input/output specs, a mapping from
    # TorchScript value names to FX nodes, converted subgraphs, and sets tracking
    # non-tensor attribute values and sub-block updates. Sub-blocks get their own
    # converter instance sharing the tables. Handler methods named
    # convert_<namespace>_<op> translate individual node kinds; the standard
    # operator handlers are attached per instance in the constructor. Key
    # method: convert, which returns the finished GraphModule.
    def __init__(
        self,
        ts_graph: torch._C.Graph | torch._C.Block,
        name_to_param: dict[str, torch.Tensor],
        name_to_buffer: dict[str, torch.Tensor],
        blocks_to_lifted_attrs: dict[torch._C.Block, set[str]],
        name_to_non_tensor_attribute: dict[str, Any],
        name_to_constant: dict[str, Any],
        name_to_attribute_fqn: dict[str, str],
    ):
        # [FUNCTION DOC] Stores the TorchScript graph/block and the shared tables
        # (the constant and FQN dictionaries are shared, not copied, with the
        # caller), creates an empty FX graph and empty spec lists, and attaches
        # one instance attribute per entry of the standard-operator table, each
        # forwarding the node to the generic standard-operator converter.
        self.ts_graph = ts_graph
        # Mapping of parameter FQN to actual parameter value
        self.name_to_param = name_to_param
        # Mapping of buffer FQN to actual buffer value
        self.name_to_buffer = name_to_buffer

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()
        self.input_specs: list[InputSpec] = []
        self.output_specs: list[OutputSpec] = []

        # Mapping of TS node name to converted FX node
        self.name_to_node: dict[
            str, torch.fx.Node | list[torch.fx.Node] | dict[Any, torch.fx.Node]
        ] = {}
        # Mapping of TS node name to constant value (int, str, TorchBind obj,
        # tensor constants ...)
        self.name_to_constant: dict[str, Any] = name_to_constant

        # Mapping from torchscript node output name to attribute fully qualified name
        self.name_to_attribute_fqn: dict[str, str] = name_to_attribute_fqn

        # Mapping from fully qualified name to real values or a fx graph node
        # During convert, this represents the current value of a non-tensor attribute
        # One use case is:
        #   def forward(self, x):
        #        c1 = self.count
        #        self.count += 1
        #        c2 = self.count
        #        return x + c1 + c2
        self.name_to_non_tensor_attribute_node: dict[str, Any] = {}

        # Mapping from fully qualified name to initial real values inputs
        # We separate it from self.name_to_non_tensor_attribute_node since
        # we need initial real value input when we construct fx.GraphModule
        self.name_to_non_tensor_attribute: dict[str, Any] = name_to_non_tensor_attribute

        self.subgraphs: dict[str, torch.fx.GraphModule] = {}

        # Mapping of block to list of attributes that need to be lifted for each
        # block
        self.blocks_to_lifted_attrs = blocks_to_lifted_attrs

        # Populate methods for the standard operators.
        for k in kind_to_standard_operators:
            handler_func_name = ir_name_to_func_name(k)
            # Create an indirect function call:
            # convert_<namespace>_<opname> --> lambda node: _convert_standard_operator(node)
            setattr(
                self,
                handler_func_name,
                lambda node: self._convert_standard_operators(node),
            )

        # This stores a list of return results that do not appear in the original TS
        # graph's outputs. The reason we maintain this is because some operations in the sub-block
        # might have inplace updates to the variable defined in the parent fx graph. After
        # the execution of that sub-block, the variable defined in the parent fx graph also
        # needs to be updated.
        self.name_update_from_subblock_to_parent: set[str] = set()

    def _is_get_attr_node(self, fqn):
        # [FUNCTION DOC] HELPER: True if fqn names a known buffer, parameter, or
        # a constant that is a TorchBind script object (all of which become FX
        # get_attr nodes), else False.
        return (
            fqn in self.name_to_buffer
            or fqn in self.name_to_param
            or (
                fqn in self.name_to_constant
                and isinstance(self.name_to_constant[fqn], torch.ScriptObject)
            )
        )

    def _convert_block_to_subgraph(self, node: torch._C.Node, arguments: list[str]):
        # [FUNCTION DOC] HELPER: converts every block of node into a separate FX
        # subgraph using a child converter. Each name in arguments becomes a
        # placeholder of the child (normalized name). Registers each subgraph on
        # this converter and adds a get_attr node for it to this FX graph.
        # Returns (list of get_attr nodes, list of child converters).
        subgraph_nodes, subgraph_converters = [], []
        for block in node.blocks():
            subgraph_converter = TS2FXGraphConverter(
                block,
                self.name_to_param,
                self.name_to_buffer,
                self.blocks_to_lifted_attrs,
                {},
                self.name_to_constant,
                self.name_to_attribute_fqn,
            )

            for block_arg in arguments:
                normalized_block_arg_name = normalize_name(block_arg)
                placeholder_node = subgraph_converter.fx_graph.placeholder(
                    normalized_block_arg_name
                )
                subgraph_converter.name_to_node[block_arg] = placeholder_node

            subgraph = subgraph_converter.convert()
            subgraph_name = self.add_subgraph(subgraph)
            subgraph_nodes.append(self.fx_graph.get_attr(subgraph_name))
            subgraph_converters.append(subgraph_converter)
        return subgraph_nodes, subgraph_converters

    def _identify_inputs_as_arguments(self, entry):
        """
        Identify inputs from the innermost sub-block. This is needed
        for nested sub-blocks when the input is hidden in the nested sub-block.
        E.g., example IR of input is hidden in the nested sub-block.
        Graph[x.1]
        %1 = ...
            Block[]
                Block[x.1]
                    %2 = x.1 ...
        """
        # [FUNCTION DOC] HELPER: recursively collects, from the blocks of entry
        # and all nested blocks, the names of node inputs that are already
        # converted in this converter (present in the value-to-node map) and are
        # not attribute values. Returns that set of names.
        arguments: set[str] = set()
        for block in entry.blocks():
            for block_node in block.nodes():
                for block_node_in in block_node.inputs():
                    if (
                        block_node_in.debugName() in self.name_to_node
                        and block_node_in.debugName() not in self.name_to_attribute_fqn
                    ):
                        arguments.add(block_node_in.debugName())
                arguments = arguments.union(
                    self._identify_inputs_as_arguments(block_node)
                )
        return arguments

    def is_top_level_graph(self):
        # [FUNCTION DOC] True if the source is a whole graph rather than a nested block.
        return isinstance(self.ts_graph, torch._C.Graph)

    def add_subgraph(self, subgraph) -> str:
        # [FUNCTION DOC] Registers subgraph under a generated name of the form
        # subgraph_N (N is the current count); mutates the subgraph table and
        # returns the name.
        name =f"subgraph_{len(self.subgraphs)}"
        self.subgraphs[name] = subgraph
        return name

    def get_args_kwargs(self, node: torch._C.Node, schema):
        # [FUNCTION DOC] Pairs node inputs with schema arguments (stops at the
        # shorter), resolves each input to an FX value, and splits them into a
        # positional tuple and a keyword dictionary according to whether the
        # schema argument is keyword-only. Returns (args tuple, kwargs dict).
        args = []
        kwargs = {}
        for input, schema_arg in zip(node.inputs(), schema.arguments):
            if schema_arg.kwarg_only:
                kwargs[schema_arg.name] = self.get_fx_value_by_ir_value(input)
            else:
                args.append(self.get_fx_value_by_ir_value(input))

        return tuple(args), kwargs

    def get_fx_value_by_ir_value(self, value: torch._C.Value):
        # [FUNCTION DOC] Resolves a TorchScript value to its FX counterpart by
        # debug name: an already converted node; else a constant (a get_attr
        # node is created for TorchBind script objects, other constants are
        # returned as plain values); else the value of the attribute FQN it maps
        # to. Raises ValueError if the name is unknown.
        value_name = value.debugName()

        if value_name in self.name_to_node:
            input_node = self.name_to_node[value_name]
            return input_node
        elif value_name in self.name_to_constant:
            if isinstance(self.name_to_constant[value_name], torch.ScriptObject):
                return self.fx_graph.get_attr(value_name)
            return self.name_to_constant[value_name]
        elif value_name in self.name_to_attribute_fqn:
            return self.get_fx_value_by_fqn(self.name_to_attribute_fqn[value_name])
        else:
            raise ValueError(f"Input {value_name} not found")

    def get_fx_value_by_fqn(self, name):
        # [FUNCTION DOC] Looks up name in priority order: converted nodes,
        # constants, current non-tensor attribute values, initial non-tensor
        # attribute values; returns the first match. Raises ValueError if absent.
        if name in self.name_to_node:
            fx_node = self.name_to_node[name]
        elif name in self.name_to_constant:
            fx_node = self.name_to_constant[name]
        elif name in self.name_to_non_tensor_attribute_node:
            fx_node = self.name_to_non_tensor_attribute_node[name]
        elif name in self.name_to_non_tensor_attribute:
            fx_node = self.name_to_non_tensor_attribute[name]
        else:
            raise ValueError(f"Attribute {name} not found")
        return fx_node

    def convert(self) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Converts the whole source graph: first the inputs, then
        # every node in order through the node dispatcher, then the outputs.
        # Wraps the FX graph in a GraphModule whose root holds the subgraphs,
        # parameters, buffers, non-tensor attributes and constants for attribute
        # lookup; applies the symbolic-size division rewrite and lints the graph.
        # Returns the GraphModule.
        self.convert_graph_inputs()

        for node in self.ts_graph.nodes():
            self.convert_node(node)

        self.convert_graph_outputs()

        # Pass parameter and buffer to the root for lookup.
        gm = torch.fx.GraphModule(
            {
                **self.subgraphs,
                **self.name_to_param,
                **self.name_to_buffer,
                **self.name_to_non_tensor_attribute,
                **self.name_to_constant,
            },
            self.fx_graph,
        )

        inplace_optimize_sym_size_div(gm)

        gm.graph.lint()

        return gm

    def convert_graph_inputs(self):
        # [FUNCTION DOC] Creates FX input nodes for each source graph input and
        # appends the matching InputSpec. Parameters and buffers become
        # parameter/buffer specs (buffers marked persistent) with get_attr nodes
        # at top level or placeholders in sub-blocks; TorchBind constants become
        # custom-object specs (AssertionError if the constant is not a script
        # object); unused class-typed inputs are skipped; everything else is a
        # user-input placeholder with a name prefixed when not a valid
        # identifier. Records each created node in the value-to-node mapping.
        for graph_input in self.ts_graph.inputs():
            name = graph_input.debugName()

            if name in self.name_to_param:
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.PARAMETER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_buffer:
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.BUFFER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                        persistent=True,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif name in self.name_to_constant:
                if not isinstance(self.name_to_constant[name], torch.ScriptObject):
                    raise AssertionError(
                        f"Input conversion only handles ScriptObject, got {type(self.name_to_constant[name])}"
                    )
                normalized_name = normalize_name(name)
                self.input_specs.append(
                    InputSpec(
                        InputKind.CUSTOM_OBJ,
                        arg=CustomObjArgument(
                            name=normalized_name, class_fqn=normalized_name
                        ),
                        target=name,
                        persistent=False,
                    )
                )
                fx_node = get_node_as_placeholder_or_get_attr(
                    self.fx_graph, name, self.is_top_level_graph()
                )
            elif isinstance(graph_input.type(), torch.ClassType):
                # Directly skip inputs that are ScriptObject but not used in the graph.
                continue
            else:
                normalized_name = normalize_name(name, prefix="input")
                self.input_specs.append(
                    InputSpec(
                        InputKind.USER_INPUT,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
                fx_node = self.fx_graph.placeholder(normalized_name)

            self.name_to_node[name] = fx_node

    def convert_aten_Float(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts a conversion-to-float op into a call of a local
        # function that casts a tensor to the default float dtype and extracts
        # its Python scalar; the node inputs are passed as arguments. Records the
        # result under the node's output name.
        def to_float_tensor(t):
            # [FUNCTION DOC] HELPER: casts tensor t to float32 and returns the scalar value.
            return t.to(dtype=torch.float).item()

        inp_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]
        fx_node = self.fx_graph.call_function(
            to_float_tensor,
            tuple(inp_list),
        )
        self.name_to_node[node.output().debugName()] = fx_node

    def convert_aten_tensor(self, node: torch._C.Node):
        """aten::tensor creates a constant tensor ad-hoc --> GetAttr"""
        # [FUNCTION DOC] Converts tensor-construction from data into a call of a
        # local target function. Arguments are resolved against the tensor
        # schema; the requires_grad keyword is coerced from an integer to a bool.
        # The construction function is the plain tensor constructor when all
        # positional args are ints, otherwise the reference implementation. The
        # target translates an integer dtype code to a real dtype before calling.
        # Records the call node under the output name.
        args, kwargs = self.get_args_kwargs(node, torch.ops.aten.tensor.default._schema)

        for k in kwargs:
            if k == "requires_grad":
                kwargs[k] = bool(kwargs[k])  # 0 -> False, 1 -> True

        to_tensor = (
            torch.tensor
            if all(isinstance(a, int) for a in args)
            else torch._refs.tensor
        )

        def target(*args, **kwargs):
            # [FUNCTION DOC] HELPER: maps an integer dtype code in kwargs (when
            # present and not None) to a torch dtype, mutating kwargs, then calls
            # the selected tensor constructor and returns the tensor.
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                kwargs["dtype"] = _TORCH_ENUM_TO_DTYPE[kwargs["dtype"]]
            return to_tensor(*args, **kwargs)

        # def to_dynamic_tensor(*args, **kwargs):
        #     if "dtype" in kwargs and kwargs["dtype"] is not None:
        #         kwargs["dtype"] = _TORCH_ENUM_TO_DTYPE[kwargs["dtype"]]
        #     return torch._refs.tensor(*args, **kwargs)

        output_name = node.output().debugName()
        fx_node = self.fx_graph.call_function(target, args, kwargs)
        self.name_to_node[output_name] = fx_node

    def convert_aten_append(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts list append into a functional call that builds
        # a new list, and rebinds BOTH the node output name and the original list
        # input name to the new node (emulating in-place mutation). Emits a
        # warning because conversion becomes order dependent. In a sub-block, if
        # the list is a placeholder, remembers its name so the parent graph gets
        # the update.
        # special handle python list append: "aten::append.t(t[](a!) self, t(c -> *) el) -> t[](a!)"

        # inplace append to the list!! This is kinda crazy, as we are inplace mutating the list
        # This makes the converter "non-functional", and the result depends on the order of the nodes being converted
        # In a sense, the converter now becomes a stateful interpreter
        warnings.warn(
            "Converting aten::append.t, which is an inplace mutation of the list. "
            "This makes the converter non-functional: the result depends on the order of the append nodes being converted!",
            stacklevel=2,
        )

        args = tuple(self.get_fx_value_by_ir_value(inp) for inp in node.inputs())
        fx_node = self.fx_graph.call_function(list_append, args)
        self.name_to_node[node.output().debugName()] = fx_node

        # inplace mutate arg[0], which is the python list
        self.name_to_node[node.inputsAt(0).debugName()] = fx_node

        # Variables that need to be updated to parent module.
        if not self.is_top_level_graph() and args[0].op == "placeholder":
            self.name_update_from_subblock_to_parent.add(node.inputsAt(0).debugName())

    def convert_prim_Constant(self, node: torch._C.Node):
        # [FUNCTION DOC] Records the constant value of the node in the constant
        # table under its output name. Supports int, float, string, interpreter
        # value kinds and None (no value attribute). A tensor constant is stored
        # under a lifted-tensor alias name and gets a get_attr node mapped to the
        # original output name. Raises ValueError for other kinds.
        name =node.output().debugName()

        value: Any = None
        if node.hasAttribute("value"):
            constant_kind = node.kindOf("value")
            if constant_kind == "i":
                value = node.i("value")
            elif constant_kind == "f":
                value = node.f("value")
            elif constant_kind == "s":
                value = node.s("value")
            elif constant_kind == "t":
                alias_name = (
                    f"lifted_tensor_{name}"  # Follow naming convention from EP tracing.
                )
                fx_node = self.fx_graph.get_attr(alias_name)
                self.name_to_node[name] = fx_node
                name, value = alias_name, node.t("value")
            elif constant_kind == "ival":
                value = node.ival("value")
            else:
                raise ValueError(f"Unsupported constant type: {node.kindOf('value')}")
        else:
            value = None

        self.name_to_constant[name] = value

    def convert_prim_CallMethod(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts a method call into an FX call_method node using
        # the node's name attribute as method name and all node inputs (the first
        # being the receiver) as arguments; records it under the output name.
        inp_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]
        fx_node = self.fx_graph.call_method(
            node.s("name"),
            tuple(inp_list),
        )
        self.name_to_node[node.output().debugName()] = fx_node

    def convert_prim_device(self, node: torch._C.Node):
        # [FUNCTION DOC] Stores the device of the tensor-typed input as a constant
        # under the output name. Raises ValueError if the input is not a tensor type.
        input_type = node.input().type()
        if input_type.isSubtypeOf(torch._C.TensorType.get()):
            device = input_type.device()  # type: ignore[attr-defined]
            output_name = node.output().debugName()
            self.name_to_constant[output_name] = device
        else:
            raise ValueError(f"Unsupported JitType ({input_type}) when get device")

    def convert_prim_GetAttr(self, node: torch._C.Node):
        # [FUNCTION DOC] Handles an attribute read. Computes the attribute FQN and
        # records it for the output name. At top level: a parameter, buffer or
        # script object gets a get_attr node; a non-tensor attribute maps to its
        # current tracked value (seeded from the initial values on first use). In
        # a sub-block: parameters/buffers/script objects reuse the already lifted
        # node; other attributes get no mapping.
        # Build fully qualified name
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = attr_fqn

        if self.is_top_level_graph():
            if self._is_get_attr_node(attr_fqn):
                # We insert a get_attr node due to two reasons.
                # First, ts graph does not lift tensor constants as input nodes. So
                # tensor constants may be ignored by in convert_graph_inputs().
                # Second, attr_fqn may have been written to via SetAttr. Two
                # GetAttr may give different values.
                self.name_to_node[output_name] = self.fx_graph.get_attr(attr_fqn)
            else:
                if attr_fqn not in self.name_to_non_tensor_attribute_node:
                    self.name_to_non_tensor_attribute_node[attr_fqn] = (
                        self.name_to_non_tensor_attribute[attr_fqn]
                    )
                self.name_to_node[output_name] = self.name_to_non_tensor_attribute_node[
                    attr_fqn
                ]
        else:
            # Special support for if blocks which do not allow SetAttr TorchScript
            # node and get_attr FX Graph Node.
            if self._is_get_attr_node(attr_fqn):
                self.name_to_node[output_name] = self.name_to_node[attr_fqn]

    def convert_prim_SetAttr(self, node: torch._C.Node):
        # [FUNCTION DOC] Handles an attribute write. For a parameter, buffer or
        # script object, emits a get_attr node plus an in-place copy of the new
        # value into it; for non-tensor attributes, just updates the tracked
        # current value for the FQN. No output is recorded.
        attr_fqn = get_attribute_fqn_from_ts_node(self.name_to_attribute_fqn, node)
        attr_value = tuple(node.inputs())[1]
        ts_graph_tensor_input = self.get_fx_value_by_ir_value(attr_value)
        if self._is_get_attr_node(attr_fqn):
            fx_attr_node = self.fx_graph.get_attr(attr_fqn)
            self.fx_graph.call_function(
                torch.Tensor.copy_, (fx_attr_node, ts_graph_tensor_input)
            )
        else:
            self.name_to_non_tensor_attribute_node[attr_fqn] = ts_graph_tensor_input

    def convert_call_function_op(self, node: torch._C.Node):
        # [FUNCTION DOC] Generic conversion of an ATen op node: resolves the op
        # overload from the node schema, builds args/kwargs, and emits a
        # call_function node. A single-output node maps its output name to the
        # call; a multi-output node maps each output name to an index-extraction
        # node on the call result.
        target = get_op_overload(node)

        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # TODO: convert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        if node.outputsSize() == 1:
            output_name = node.output().debugName()
            self.name_to_node[output_name] = fx_node
        else:
            for i, outp in enumerate(node.outputs()):
                output_name = outp.debugName()
                next_fx_node = self.fx_graph.call_function(
                    operator.getitem, (fx_node, i)
                )
                self.name_to_node[output_name] = next_fx_node

    def convert_prim_TupleConstruct(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts tuple construction by delegating to the shared
        # iterator conversion (result is a Python list of values).
        self._convert_prim_iterator(node)

    def convert_prim_ListConstruct(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts list construction by delegating to the shared
        # iterator conversion.
        self._convert_prim_iterator(node)

    def _convert_prim_iterator(self, node: torch._C.Node):
        # [FUNCTION DOC] HELPER: resolves all inputs to FX values and stores them
        # as a plain Python list (no FX node emitted) under the output name.
        output_list = [self.get_fx_value_by_ir_value(inp) for inp in node.inputs()]

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_list

    def convert_prim_DictConstruct(self, node: torch._C.Node):
        # [FUNCTION DOC] Builds a Python dict from inputs taken pairwise (key,
        # value, key, value, ...) with FX values resolved, and stores it under the
        # output name. Raises AssertionError on a missing/None key or value or an
        # odd number of inputs.
        output_dict = {}
        k, v = None, None
        for i, inp in enumerate(node.inputs()):
            # We assume key value are stored in pair in the DictConstruct.
            # The first element is the key and the following is the value.
            if i % 2 == 0:
                k = self.get_fx_value_by_ir_value(inp)
            else:
                v = self.get_fx_value_by_ir_value(inp)
                if k is None or v is None:
                    raise AssertionError("DictConstruct has an empty key value pair.")
                output_dict[k] = v
                k, v = None, None

        if k is not None or v is not None:
            raise AssertionError(
                "DictConstruct has an odd number of elements (violating our assumption)."
            )

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_dict

    def convert_prim_ListUnpack(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts list unpacking via the shared unpack helper.
        self._convert_prim_unpack_iterator(node)

    def convert_prim_TupleUnpack(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts tuple unpacking via the shared unpack helper.
        self._convert_prim_unpack_iterator(node)

    def _convert_prim_unpack_iterator(self, node: torch._C.Node):
        # [FUNCTION DOC] HELPER: for each output of the node, emits an
        # index-extraction call on the single input container (index equals the
        # output position) and maps the output name to it.
        # Single input and multiple outputs for unpacking.
        for i, outp in enumerate(node.outputs()):
            outp_name = outp.debugName()
            inp = self.get_fx_value_by_ir_value(node.input())
            fx_node = self.fx_graph.call_function(operator.getitem, (inp, i))
            self.name_to_node[outp_name] = fx_node

    def convert_aten_Int(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts integer conversion of a tensor into a copy cast
        # to int32 followed by extraction of the scalar value; records the scalar
        # node under the output name.
        # converts aten::Int as aten._to_copy + aten::_local_scalar_dense
        target = torch.ops.aten._to_copy.default
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        to_copy_node = self.fx_graph.call_function(target, args, {"dtype": torch.int32})

        fx_node = self.fx_graph.call_function(
            torch.ops.aten._local_scalar_dense.default, (to_copy_node,)
        )

        # TODO: convert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_NumToTensor(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts number-to-tensor into a scalar_tensor call with
        # dtype int64, using the node inputs as arguments; records it under the
        # output name.
        # Converts prim::NumToTensor as aten.scalar_tensor.
        # prim::NumToTensor IRs are currently triggered by:
        # .size() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L950
        # .numel() https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/frontend/tracer.cpp#L971
        # For both of those APIs, torch.jit.trace implicitly sets the output tensor type
        # to be LongTensor.
        target = torch.ops.aten.scalar_tensor
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())

        fx_node = self.fx_graph.call_function(target, args, {"dtype": torch.long})
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_CreateObject(self, node: torch._C.Node):
        # [FUNCTION DOC] Registers the created object's output name with an empty
        # attribute FQN (the root of subsequent attribute paths). Emits no FX node.
        output_name = node.output().debugName()
        self.name_to_attribute_fqn[output_name] = ""

    def convert_aten__convolution(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts the internal convolution op into the public
        # convolution op, matching its schema arguments; records the call under
        # the output name.
        # converts aten::_convolution as aten.convolution, since aten::_convolution
        # doesn't have a meta function
        target = torch.ops.aten.convolution.default
        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_div(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts division. For the tensor-with-rounding-mode
        # overload whose second argument is a single-element tensor constant,
        # emits the scalar-mode division with the constant's Python value.
        # Otherwise falls back to the generic op conversion.
        # KNOWN ISSUE: the second argument's FX node name is looked up in the
        # constant table keyed by TorchScript names, and a non-node argument
        # (for example a plain number) has no name attribute and would raise.
        target = get_op_overload(node)
        schema = target._schema

        args, kwargs = self.get_args_kwargs(node, schema)

        # converts aten::div.Tensor_mode(x, tensor_constant)
        # as aten.div.Scalar_mode(x, tensor_constant.item())
        if schema.overload_name == "Tensor_mode":
            arg1_name = args[1].name
            if arg1_name in self.name_to_constant and isinstance(
                self.name_to_constant[arg1_name], torch.Tensor
            ):
                tensor_constant = self.name_to_constant[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.name_to_constant[arg1_name].item()

                    fx_node = self.fx_graph.call_function(
                        torch.ops.aten.div.Scalar_mode,
                        tuple(updated_args),
                        kwargs,
                    )

                    # TODO: convert sourceRange() into stack_trace
                    # fx_node.meta["stack_trace"] = node.sourceRange()

                    output_name = node.output().debugName()
                    self.name_to_node[output_name] = fx_node
                    return

        self.convert_call_function_op(node)

    def convert_aten___getitem__(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts container indexing into an index-extraction call
        # on the container with the given index; records it under the output name.
        input_container, index = tuple(
            self.get_fx_value_by_ir_value(input) for input in node.inputs()
        )
        fx_node = self.fx_graph.call_function(
            operator.getitem, (input_container, index)
        )
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_to(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts the dtype-conversion op. If it is the dtype or
        # prim_dtype overload and any user of its output is a mutating op, emits
        # the dtype overload with the copy flag forced to True followed by a
        # clone (workaround for functionalization errors on frozen storage) and
        # maps the output to the clone; AssertionError if fewer than 4 args.
        # Otherwise uses the generic op conversion.
        target = get_op_overload(node)
        args, _kwargs = self.get_args_kwargs(node, target._schema)

        # special handle aten.to.dtype and aten.to.prim_dtype followed by inplace_mutation_op
        # coz aten.to + inplace_mutation_op pattern would trigger
        # "cannot mutate tensors with frozen storage" functionalization error.
        # To work around the issue, we override the copy to be True, so that the output
        # is for sure not an alias of input
        if target is torch.ops.aten.to.dtype or target is torch.ops.aten.to.prim_dtype:
            user_nodes = [use.user for use in node.output().uses()]
            user_targets = [
                get_op_overload(user_node)
                for user_node in user_nodes
                if user_node.schema() != "(no schema)"
            ]
            has_mutable_target = any(
                target._schema.is_mutable for target in user_targets
            )

            if has_mutable_target:
                if len(args) < 4:
                    raise AssertionError(f"expected at least 4 args, got {len(args)}")
                new_args = list(args)
                new_args[3] = True  # copy, override to True
                fx_node = self.fx_graph.call_function(
                    torch.ops.aten.to.dtype, tuple(new_args)
                )
                # temp hack to work around the issue https://github.com/pytorch/pytorch/issues/131679
                # When this issue is fixed, the clone node would be no longer needed
                clone_node = self.fx_graph.call_function(
                    torch.ops.aten.clone.default, (fx_node,)
                )
                output_name = node.output().debugName()
                self.name_to_node[output_name] = clone_node
                return

        self.convert_call_function_op(node)

    def convert_aten_add(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts addition. A node without a schema is accepted
        # only when both inputs are lists (treated as list concatenation), else
        # RuntimeError. List add is emitted as a call to the list_add helper;
        # every other overload uses the generic op conversion.
        if node.schema() == "(no schema)":
            if isinstance(node.inputsAt(0).type(), torch.ListType) and isinstance(
                node.inputsAt(1).type(), torch.ListType
            ):
                target = torch.ops.aten.add.t
            else:
                raise RuntimeError(f"unable to determined the target for {node}")
        else:
            target = get_op_overload(node)

        if target is torch.ops.aten.add.t:
            # special handle python list/tuple add: "aten::add.t(t[] a, t[] b) -> t[]" for
            # RuntimeError: aten::add() Expected a value of type 'List[t]' for argument 'a' but instead found type 'immutable_list'.
            args, _kwargs = self.get_args_kwargs(node, target._schema)
            output_name = node.output().debugName()
            self.name_to_node[output_name] = self.fx_graph.call_function(list_add, args)
        else:
            self.convert_call_function_op(node)

    def _check_prim_loop_support(self, node):
        # [FUNCTION DOC] HELPER: validates a loop node is convertible. Raises
        # RuntimeError if the iteration count input is not a known constant, or if
        # the loop condition returned by the body is produced by any node inside
        # the body (i.e. changes dynamically).
        inputs = list(node.inputs())

        # TODO: (1/N) stage.
        if inputs[0].debugName() not in self.name_to_constant:
            raise RuntimeError(
                "prim::Loop currently cannot run with dynamic value of number of iterations."
            )

        # Make sure the condition is not updated in the subblock.
        subblock = next(node.blocks())
        condition_output_name = next(subblock.outputs()).debugName()
        for node in subblock.nodes():
            if (
                node.outputsSize() == 1
                and node.output().debugName() == condition_output_name
            ):
                raise RuntimeError(
                    "prim::Loop currently cannot run with dynamic value of condition."
                )
            if node.outputsSize() >= 2:
                for outp in node.outputs():
                    if outp.debugName() == condition_output_name:
                        raise RuntimeError(
                            "prim::Loop currently cannot run with dynamic value of condition."
                        )

    def convert_prim_Loop(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts a loop with a constant trip count by unrolling:
        # validates support, converts the body block into one subgraph taking
        # captured outer values (inputs from nested blocks plus lifted
        # attributes), then emits one call of the loop-body executor per
        # iteration. After each call, loop outputs (index shifted by one because
        # element zero is the condition) and in-place-updated outer variables are
        # rebound to new extraction nodes so the next iteration sees them.
        # Propagates sub-block update names to the parent when nested.
        # KNOWN ISSUE: the position used to rebind an in-place-updated captured
        # value in the argument list is computed from the loop output count plus
        # the iteration order of the update set, not from the number of
        # loop-carried inputs, so it can point at the wrong argument.
        inputs = list(node.inputs())
        self._check_prim_loop_support(node)

        num_iterations = self.get_fx_value_by_ir_value(inputs[0])

        # Find inputs.
        loop_local_arguments = [inp.debugName() for inp in inputs[2:]]

        global_arguments = self._identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            global_arguments = global_arguments.union(
                self.blocks_to_lifted_attrs[block]
            )

        global_arguments = list(global_arguments)

        subgraph_nodes, subgraph_converters = self._convert_block_to_subgraph(
            node, global_arguments
        )

        if len(subgraph_nodes) != 1:
            raise AssertionError(f"expected 1 subgraph node, got {len(subgraph_nodes)}")
        subgraph_converter = subgraph_converters[0]
        if not self.is_top_level_graph():
            self.name_update_from_subblock_to_parent = (
                self.name_update_from_subblock_to_parent.union(
                    subgraph_converter.name_update_from_subblock_to_parent
                )
            )

        fx_block_args = [
            self.get_fx_value_by_fqn(name)
            for name in loop_local_arguments + global_arguments
        ]
        for iter_idx in range(num_iterations):
            loop_node = self.fx_graph.call_function(
                execute_subgraph_from_prim_loop,
                # Check execute_node function for the expected arguments order.
                (
                    subgraph_nodes[0],
                    iter_idx,
                    len(loop_local_arguments),
                    *fx_block_args,
                ),
                {},
            )

            # Update the value of loop local variables.
            if node.outputsSize() >= 1:
                for i, outp in enumerate(node.outputs()):
                    output_name = outp.debugName()
                    self.name_to_node[output_name] = self.fx_graph.call_function(
                        operator.getitem,
                        (
                            loop_node,
                            i + 1,
                        ),  # + 1 because the 0th element is the condition.
                    )
                    fx_block_args[i] = self.name_to_node[output_name]

            # Update the value of global variables, whose values are modified inplace.

            for i, name in enumerate(
                subgraph_converter.name_update_from_subblock_to_parent
            ):
                self.name_to_node[name] = self.fx_graph.call_function(
                    operator.getitem,
                    (
                        loop_node,
                        i + node.outputsSize() + 1,
                    ),  # + 1 because the 0th element is the condition.
                )
                global_argument_index = global_arguments.index(name)
                fx_block_args[i + node.outputsSize() + global_argument_index] = (
                    self.name_to_node[name]
                )

    def _check_set_attr_in_if_block(self, if_node: torch._C.Node):
        # [FUNCTION DOC] HELPER: scans the top-level nodes of each branch block of
        # the conditional and raises RuntimeError if any attribute-write node is
        # present (unsupported when converting to a cond). Nested blocks are not scanned.
        for block in if_node.blocks():
            for node in block.nodes():
                if node.kind() == "prim::SetAttr":
                    raise RuntimeError(
                        "During converting prim::If to torch.cond, found prim::SetAttr op"
                        " which is not supported yet. Please file an issue if you come "
                        "across this error."
                    )

    def convert_prim_If(self, node: torch._C.Node):
        # [FUNCTION DOC] Converts a conditional into a torch.cond call. Rejects
        # branches containing attribute writes, requires exactly one input (the
        # predicate, AssertionError otherwise), converts both blocks into
        # subgraphs taking the same operands (values used inside nested blocks
        # plus lifted attributes), and emits cond with predicate, true graph,
        # false graph and operand tuple. Zero outputs map nothing; one output maps
        # to the cond result; several outputs map to index extractions of it.
        self._check_set_attr_in_if_block(node)

        inputs = list(node.inputs())
        if len(inputs) != 1:
            raise AssertionError(f"expected 1 input for prim::If, got {len(inputs)}")
        predicate = self.get_fx_value_by_ir_value(inputs[0])

        # Find inputs.
        arguments = self._identify_inputs_as_arguments(node)

        # Lift parameters as inputs.
        for block in node.blocks():
            arguments = arguments.union(self.blocks_to_lifted_attrs[block])

        arguments = list(arguments)
        subgraph_nodes, _ = self._convert_block_to_subgraph(node, arguments)

        if len(subgraph_nodes) != 2:
            raise AssertionError(
                f"expected 2 subgraph nodes, got {len(subgraph_nodes)}"
            )

        fx_block_args = [self.get_fx_value_by_fqn(name) for name in arguments]

        args = (
            predicate,
            subgraph_nodes[0],
            subgraph_nodes[1],
            tuple(fx_block_args),
        )

        cond_node = self.fx_graph.call_function(torch.cond, args, {})

        # prim::If can also have zero output.
        if node.outputsSize() == 1:
            output_name = node.output().debugName()
            self.name_to_node[output_name] = cond_node
        elif node.outputsSize() > 1:
            for i, output in enumerate(node.outputs()):
                output_name = output.debugName()
                getitem = self.fx_graph.call_function(operator.getitem, (cond_node, i))
                self.name_to_node[output_name] = getitem

    def convert_aten_Bool(self, node: torch._C.Node):
        # [FUNCTION DOC] Treats the bool conversion as a no-op alias of its first argument.
        self._convert_as_noop(node)

    def convert_prim_Enter(self, node: torch._C.Node):
        # [FUNCTION DOC] Context-manager entry is ignored; emits nothing.
        # export generally treats prim::Enter as noop
        # The only context manager export supports is aten::enable_grad.
        # Unfortunately, TorchScript does not support aten::enable_grad yet.
        # TODO: support aten::enable_grad in both TorchScript and Converter.
        return

    def convert_prim_Exit(self, node: torch._C.Node):
        # [FUNCTION DOC] Context-manager exit is ignored; emits nothing.
        # export treats prim::Exit as noop
        return

    def _convert_as_noop(self, node: torch._C.Node):
        # [FUNCTION DOC] HELPER: resolves the op overload and its arguments, then
        # maps the node output name to the first argument itself (no FX node emitted).
        # Converts the node as a no-op by mapping its output node as arg[0]

        target = get_op_overload(node)
        schema = target._schema

        args, _kwargs = self.get_args_kwargs(node, schema)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = args[0]

    def convert_profiler__record_function_exit(self, node: torch._C.Node):
        # [FUNCTION DOC] Emits a call to the profiler record-function-exit op with
        # the node inputs, kept in the graph for its side effect. Output not recorded.
        # _record_function_exit has side effect so we keep it in fx.graph
        # currently, _record_function_enter_new and _record_function_exit are
        # discarded during `retrace_as_exported_program`.
        target = torch.ops.profiler._record_function_exit
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        self.fx_graph.call_function(target, args)

    def convert_prim_tolist(self, node: torch._C.Node):
        # [FUNCTION DOC] Emits a method call to the tensor method that converts to
        # a Python list on the first input; records it under the output name.
        # prim::tolist cannot be supported by `_convert_standard_operators`
        # since it requires call_method instead of call_function.
        target = "tolist"
        args = (self.get_fx_value_by_ir_value(next(node.inputs())),)
        fx_node = self.fx_graph.call_method(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_Uninitialized(self, node: torch._C.Node):
        # [FUNCTION DOC] Registers an empty tensor as a dummy constant for the
        # output name (the value is never actually used).
        # `prim::Uninitialized` is inserted by the compiler when it can prove
        # the value will never be used. It can be introduced by exceptions,
        # breaks, continues, and returns.
        # So we add a dummy constant to the graph.
        output_name = node.output().debugName()
        self.name_to_constant[output_name] = torch.Tensor()

    def _convert_standard_operators(self, node: torch._C.Node):
        # [FUNCTION DOC] HELPER: shared handler for node kinds in the
        # standard-operator table. Looks up the Python/ATen callable for the node
        # kind, emits a call with all inputs resolved, and records it under the
        # single output name.
        target = kind_to_standard_operators[node.kind()]
        args = tuple(self.get_fx_value_by_ir_value(input) for input in node.inputs())
        fx_node = self.fx_graph.call_function(target, args)
        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_node(self, node: torch._C.Node):
        # [FUNCTION DOC] Dispatches one node: finds the handler method by kind
        # name (falling back to the generic op conversion), logs the first line
        # of the node text at debug level, and calls the handler. Any exception
        # is re-raised as RuntimeError naming the node kind, chaining the cause.
        node_kind = node.kind()

        # Get handler based on namespace and operator name.
        # Provide a default node handler as well in case we don't find
        # matching converter for that.
        handler_func_name = ir_name_to_func_name(node_kind)
        handler_func = getattr(self, handler_func_name, self.convert_call_function_op)

        # str calls print function implemented in CPP. To avoid repeating
        # the entire logic here, we simply keep first line from node string (getting rid
        # of sub-blocks IR prints).
        node_str = "".join(str(node).split("\n")[:1])
        log.debug("[%s] converts [%s]", handler_func.__name__, node_str)
        try:
            handler_func(node)
        except Exception as e:
            raise RuntimeError(f"TS2EPConverter failed for node {node_kind}") from e

    def convert_graph_outputs(self):
        # [FUNCTION DOC] Emits the FX output for the source graph outputs plus
        # names updated in sub-blocks that must propagate to the parent. Each name
        # resolves to a converted node (a sub-block output that is a bare
        # placeholder is cloned) or a constant; an OutputSpec of user-output kind
        # is appended for each. Raises ValueError for an unknown name. The graph
        # output is empty, a single value, or a list depending on the count.
        # KNOWN ISSUE: the final else branch is unreachable because the count
        # cannot be negative.
        args = []
        outp_name_list = [outp.debugName() for outp in self.ts_graph.outputs()] + list(
            self.name_update_from_subblock_to_parent
        )
        for output_name in outp_name_list:
            if output_name in self.name_to_node:
                fx_node = self.name_to_node[output_name]
                # TODO: Revisit this later after HigherOrderOp design changes.
                # Currently, we cannot directly return input as output.
                if (
                    not self.is_top_level_graph()
                    and isinstance(fx_node, torch.fx.Node)
                    and fx_node.op == "placeholder"
                ):
                    fx_node = self.fx_graph.call_function(torch.clone, (fx_node,))
                args.append(fx_node)
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=output_name),
                        target=output_name,
                    )
                )
            elif output_name in self.name_to_constant:
                args.append(self.name_to_constant[output_name])
                self.output_specs.append(
                    OutputSpec(
                        OutputKind.USER_OUTPUT,
                        arg=ConstantArgument(
                            name=output_name, value=self.name_to_constant[output_name]
                        ),
                        target=output_name,
                    )
                )
            else:
                raise ValueError(f"Output {output_name} not found")

        if len(args) == 0:
            # Sub-block of prim::If can have zero output.
            self.fx_graph.output([])
        elif len(args) == 1:
            self.fx_graph.output(
                args[0]
            )  # Get rid of an extra list wrapped around final output.
        elif len(args) > 1:
            self.fx_graph.output(
                args
            )  # For prim::Loop and prim::If with multiple outputs.
        else:
            # Sub-block of prim::Loop can have multiple outputs.
            self.fx_graph.output(args)


class ExplainTS2FXGraphConverter(TS2FXGraphConverter):
    """
    Run TS2FXGraphConverter in an explain mode. It collects all failed operators conversions
    and provide that information to users. In order to collect all failed conversions, it
    also mocks some internal attributes (e.g., name_to_node).
    """
    # [CLASS DOC] Diagnostic variant of the graph converter. Instead of failing
    # on the first unconvertible node it records the node in
    # unsupported_node_list and continues; the value-to-node mapping is replaced
    # by a mock dictionary that answers every lookup (with a dummy node) so later
    # nodes depending on failed ones still convert. Key method: explain.

    class _DictMock(dict):
        # [CLASS DOC] HELPER: dict subclass that claims to contain every key and
        # returns a fixed mock value for keys it does not actually hold.
        def __init__(self, dict_data, mock_value):
            # [FUNCTION DOC] HELPER: copies dict_data and stores mock_value as the default.
            super().__init__(dict_data)
            self.mock_value = mock_value

        def __getitem__(self, key):
            # [FUNCTION DOC] HELPER: returns the stored value for key if present,
            # else the mock value (never raises KeyError).
            # If the original dictionary has the key, return its value.
            # Otherwise, return the mock value.
            if not super().__contains__(key):
                return self.mock_value
            return super().__getitem__(key)

        def __contains__(self, key):
            # [FUNCTION DOC] HELPER: always True, so every name appears defined.
            return True

    def __init__(
        self,
        ts_graph: torch._C.Graph | torch._C.Block,
        name_to_param: dict[str, torch.Tensor],
        name_to_buffer: dict[str, torch.Tensor],
        blocks_to_lifted_attrs: dict[torch._C.Block, set[str]],
        name_to_non_tensor_attribute: dict[str, Any],
        name_to_constant: dict[str, Any],
        name_to_attribute_fqn: dict[str, str],
    ):
        # [FUNCTION DOC] Initializes the base converter with the same arguments,
        # creates an empty list of unsupported nodes, and wraps the value-to-node
        # mapping in the mock dictionary whose default is a dummy call_function node.
        super().__init__(
            ts_graph,
            name_to_param,
            name_to_buffer,
            blocks_to_lifted_attrs,
            name_to_non_tensor_attribute,
            name_to_constant,
            name_to_attribute_fqn,
        )

        # Data to keep track of unsupported nodes.
        self.unsupported_node_list: list[torch._C.Node] = []

        # Add mock to needed attributes.
        self.name_to_node = ExplainTS2FXGraphConverter._DictMock(
            self.name_to_node,
            # Dummy node.
            torch.fx.Node(
                None,  # type: ignore[arg-type]
                "mock",
                "call_function",
                lambda: None,
                (),
                {},
            ),
        )

    def explain(self):
        # [FUNCTION DOC] Runs conversion of inputs, all nodes and outputs without
        # building a GraphModule; failures are collected, not raised (except from
        # the output step). Returns None.
        self.convert_graph_inputs()
        for node in self.ts_graph.nodes():
            self.convert_node(node)
        self.convert_graph_outputs()

    def convert_node(self, node):
        # [FUNCTION DOC] Converts a node via the base class; if any exception
        # occurs, swallows it and appends the node to the unsupported list.
        try:
            super().convert_node(node)
        except Exception:
            self.unsupported_node_list.append(node)


@contextmanager
def disable_logging(log):
    # [FUNCTION DOC] Context manager (also usable as a decorator) that sets the
    # given logger's disabled flag for its duration and restores the previous
    # value on exit, even on error.
    disabled = log.disabled
    log.disabled = True
    try:
        yield
    finally:
        log.disabled = disabled


class TS2EPConverter:
    # [CLASS DOC] Top-level converter from a TorchScript module or function to an
    # ExportedProgram. State: the model, its jit graph and params, sample inputs,
    # and tables of parameters, buffers, non-tensor attributes and constants
    # (filled at construction, including lifted attributes). Key methods:
    # convert (build the FX graph module and retrace it as an ExportedProgram),
    # explain (report unsupported nodes) and retrace_as_exported_program.
    # TorchScript model to ExportedProgram converter
    def __init__(
        self,
        ts_model: torch.jit.ScriptModule | torch.jit.ScriptFunction,
        sample_args: tuple[Any, ...],
        sample_kwargs: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC] Builds the jit graph for ts_model with sample_args and
        # classifies every state-dict entry as a parameter (it equals some
        # parameter of the same shape) or a buffer; script functions have none.
        # Then lifts tensor, script-object and other attributes found in the
        # graph via lift_get_attr. Side effect: may trace/freeze the model.
        # KNOWN ISSUE: sample_kwargs is stored but never used by the conversion.
        self.ts_model = ts_model
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)

        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs

        self.name_to_param: dict[str, torch.Tensor] = {}
        self.name_to_buffer: dict[str, torch.Tensor] = {}
        param_list = (
            list(self.ts_model.parameters())
            if not isinstance(self.ts_model, torch._C.ScriptFunction)
            else []
        )
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                # Check if tensor belongs to any parameter.
                if any(
                    (tensor == param).all()
                    for param in param_list
                    if tensor.shape == param.shape
                ):
                    self.name_to_param[k] = tensor
                else:
                    self.name_to_buffer[k] = tensor

        self.name_to_non_tensor_attributes: dict[str, Any] = {}
        self.name_to_constant: dict[str, Any] = {}

        self.lift_get_attr()

    def convert(self) -> ExportedProgram:
        # [FUNCTION DOC] Full conversion. Logs the TorchScript graph, computes the
        # lifted-attribute analysis, runs TS2FXGraphConverter to get a
        # GraphModule, replaces quantized ops with standard ops, retraces it into
        # an ExportedProgram with the constants gathered during conversion, and
        # finally copies any original state-dict entries missing from the program
        # into its state dict with a warning. Returns the ExportedProgram.
        log.info(
            """
TS2EPConverter logging starts from here.

INFO: (TORCH_LOGS="export" <cmd>)
    * Log TorchScript IR.

DEBUG: (TORCH_LOGS="+export" <cmd>), additionally
    * Log conversion IR by IR in a format of [<conversion handler name>] converts [<IR>].
        """
        )
        log.info("TorchScript graph\n\n%s\n", self.ts_graph)

        blocks_to_lifted_attrs, name_to_attribute_fqn = get_block_to_lifted_attrs(
            self.ts_graph
        )

        graph_converter = TS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
            self.name_to_constant,
            name_to_attribute_fqn,
        )
        gm = graph_converter.convert()

        # Post-processing step to deal with quantized operators.
        replace_quantized_ops_with_standard_ops(gm)
        log.info("GraphModule: %s", gm.print_readable(print_output=False))

        ep = self.retrace_as_exported_program(
            gm,
            graph_converter.name_to_constant,
        )
        log.info("%s", ep)

        # Post-processing step to ensure ExportedProgram has the same state_dict as
        # the original TorchScript model. Throw warnings for additionally populated
        # state_dict entries.
        if not isinstance(self.ts_model, torch._C.ScriptFunction):
            for k, tensor in self.ts_model.state_dict().items():  # type: ignore[union-attr]
                if k not in ep.state_dict:
                    warnings.warn(
                        f"Manually populate {k} into state_dict ExportedProgram, but it is never used by the ExportedProgram.",
                        stacklevel=2,
                    )
                    ep.state_dict[k] = tensor

        return ep

    @disable_logging(log)
    def explain(self, print_output=True):
        # [FUNCTION DOC] Dry-run of the conversion with logging disabled: uses the
        # explain-mode converter to collect all nodes that fail to convert.
        # Builds a message listing each unsupported node kind and first IR line,
        # or "Success!" if none. Prints it if print_output is true; returns it.
        blocks_to_lifted_attrs, name_to_attribute_fqn = get_block_to_lifted_attrs(
            self.ts_graph
        )

        graph_converter = ExplainTS2FXGraphConverter(
            self.ts_graph,
            self.name_to_param,
            self.name_to_buffer,
            blocks_to_lifted_attrs,
            self.name_to_non_tensor_attributes,
            self.name_to_constant,
            name_to_attribute_fqn,
        )
        graph_converter.explain()
        if len(graph_converter.unsupported_node_list) > 0:
            explain_str = "Unsupported nodes are found in the following list:"
            for i, n in enumerate(graph_converter.unsupported_node_list):
                node_str = "".join(str(n).split("\n")[:1])
                explain_str += f"\n\n    {i}. {n.kind()} [{node_str}]"
        else:
            explain_str = "Success!"
        if print_output:
            print(explain_str)
        return explain_str

    def retrace_as_exported_program(
        self,
        gm: torch.fx.GraphModule,
        name_to_constant: dict[str, Any],
    ):
        # [FUNCTION DOC] Retraces GraphModule gm with the sample args through the
        # private export entry point (non-strict, pre-dispatch), marking every
        # dimension of tensor inputs as automatically dynamic. Then fixes up the
        # program: adds tensor and script-object constants to its constants,
        # removes their names from the state dict, and reclassifies buffer specs
        # that are really constant tensors (AssertionError if such a constant is
        # not a tensor). Runs the verifier and returns the ExportedProgram.
        dynamic_shapes = _tree_map_with_path(
            lambda path, x: (
                [Dim.AUTO] * x.dim() if isinstance(x, torch.Tensor) else None
            ),
            self.sample_args,
        )

        # TODO: adjust input orders to match GraphSignature convention
        ep = torch.export._trace._export(
            gm,
            self.sample_args,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            pre_dispatch=True,
        )

        # Post-processing to make sure the ExportedProgram states are correct.
        # Because during conversion, we set tensor constants as GetAttr,
        # retracing cannot recognize them as tensor constants but instead
        # treat them as buffers. We need to set them again here.
        ep._constants.update(
            {
                k: v
                for k, v in name_to_constant.items()
                if isinstance(v, (torch.Tensor, torch.ScriptObject))
            }
        )
        for k in name_to_constant:
            ep.state_dict.pop(k, None)

        for spec in ep.graph_signature.input_specs:
            # Mark as constant tensors for erroneously traced buffers.
            if spec.kind == InputKind.BUFFER and spec.target in name_to_constant:
                if not isinstance(name_to_constant[spec.target], torch.Tensor):
                    raise AssertionError(
                        f"{type(name_to_constant[spec.target])} has been erroneously marked as buffer"
                    )
                spec.kind = InputKind.CONSTANT_TENSOR
                spec.persistent = None
        ep.verifier().check(ep)

        return ep

    def lift_get_attr(self):
        # [FUNCTION DOC] Scans the whole jit graph (including sub-blocks) for
        # attribute reads and registers their values on the converter: tensors not
        # already known become buffers, TorchBind script objects become constants,
        # and anything else is stored as a non-tensor attribute (overwritten on
        # repeats). Looks values up on the model by dotted path. Mutates the
        # converter tables; returns None.
        # This function lifts multiple data types.

        #     1. Tensor constants attributes (e.g., self.data = torch.tensor([2,3]))
        #     to buffers. Currently, when there are tensor constants, export
        #     would error and ask users to register tensor constants as buffers.
        #     Since it is hard to manually do so for TorchScript models
        #     (e.g., source code is missing), this function automatically
        #     lifts tensor constants to be buffers.

        #     2. ScriptObbject to constant. It will then be converted to getattr in
        #     in the fx graph.
        #
        # This function should happen in TS2EPConverter instead of
        # TS2FXGraphConverter since it gets attributes from self.ts_model
        # which is not accessible in TS2FXGraphConverter. It is similar to where
        # we collect self.name_to_param and self.name_to_buffer.
        name_to_attribute_fqn: dict[str, str] = {}

        def get_attr(fqn: str):
            # [FUNCTION DOC] HELPER: resolves a dotted path on the model by
            # successive attribute lookups and returns the value.
            name = fqn.split(".")
            v = self.ts_model
            for n in name:
                v = getattr(v, n)
            return v

        def get_fqn(node: torch._C.Node):
            # [FUNCTION DOC] HELPER: returns the dotted path for an attribute-read
            # node from the parent's known path and the node's attribute name
            # (KeyError if the parent is unknown).
            attr_name = node.s("name")
            input_name = node.input().debugName()
            root_attr_name = name_to_attribute_fqn[input_name]
            attr_fqn = f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name
            return attr_fqn

        def _dfs_get_attr(block):
            # [FUNCTION DOC] HELPER: recursive walk of a block: object-creation
            # nodes get an empty path; attribute reads compute their path, record
            # it, and register tensors / script objects / other values on the
            # converter; recurses into nested blocks.
            for node in block.nodes():
                if node.kind() == "prim::CreateObject":
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = ""

                if node.kind() == "prim::GetAttr":
                    attr_fqn = get_fqn(node)
                    value = get_attr(attr_fqn)
                    output_name = node.output().debugName()
                    name_to_attribute_fqn[output_name] = attr_fqn
                    if isinstance(value, torch.Tensor):
                        if attr_fqn not in self.name_to_buffer:
                            # Lift tensor constants to be a buffer
                            self.name_to_buffer[attr_fqn] = value
                    elif isinstance(value, torch.ScriptObject):
                        if attr_fqn not in self.name_to_constant:
                            self.name_to_constant[attr_fqn] = value
                    else:
                        self.name_to_non_tensor_attributes[attr_fqn] = value

                for subblock in node.blocks():
                    _dfs_get_attr(subblock)

        _dfs_get_attr(self.ts_graph)
