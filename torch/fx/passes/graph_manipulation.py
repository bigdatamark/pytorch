# [MODULE DOC] Helpers for manipulating FX graph modules and estimating node memory sizes: replacing
# nodes by op and target, and computing per-node output and total byte sizes from shape-propagation
# metadata. No import-time side effects.
from typing import NamedTuple

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_arg, Node, Target
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata


__all__ = [
    "replace_target_nodes_with",
    "size_bytes",
    "get_size_of_all_nodes",
    "get_tensor_meta",
    "get_size_of_node",
]


@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
) -> None:
    # [FUNCTION DOC] Rebuilds the graph of the module in a new graph, copying all nodes but recreating
    # each node matching both the old op and old target with the new op and target (same remapped args,
    # kwargs and name), then assigns the new graph to the module. Mutates the module; returns nothing.
    # Raises AssertionError if the remapped args or kwargs have unexpected container types.
    # KNOWN ISSUE: node metadata of the replaced nodes is not copied to the recreated nodes.
    """
    Modifies all nodes in fx_module.graph.nodes which match the specified op code
    and target, and updates them to match the new op code and target.
    """
    new_graph = Graph()
    val_map: dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            if not isinstance(args, tuple):
                raise AssertionError(f"Expected tuple, got {type(args)}")
            if not isinstance(kwargs, dict):
                raise AssertionError(f"Expected dict, got {type(kwargs)}")
            val_map[node] = new_graph.create_node(
                new_op, new_target, args, kwargs, node.name
            )
        else:
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    fx_module.graph = new_graph


@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    # [CLASS DOC] Named pair of byte sizes: output_size is the size of the node result and total_size
    # additionally includes parameter storage of a called submodule.
    output_size: int
    total_size: int


@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(
    fx_module: GraphModule, args: list[torch.Tensor] | None = None
) -> None:
    # [FUNCTION DOC] Optionally runs shape propagation with the given example arguments, then walks the
    # nodes in order until the output node and sets a size_bytes attribute on each node using
    # get_size_of_node. Mutates nodes; returns nothing. Raises RuntimeError if tensor metadata is missing.
    # KNOWN ISSUE: the docstring claims a total size is returned, but the function returns nothing.
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    if args is not None:
        # Mark shape and dtype for each node (node.shape and node.dtype)
        ShapeProp(fx_module).propagate(*args)
    # Calculate the total size of the whole fx graph
    for node in fx_module.graph.nodes:
        if node.op == "output":
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return


@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> TensorMetadata:
    # [FUNCTION DOC] Returns the tensor_meta metadata entry of the node. Raises RuntimeError when it is
    # missing or empty, which means shape propagation has not run on the node.
    tensor_meta = node.meta.get("tensor_meta")

    if not tensor_meta:
        raise RuntimeError(
            f"Node {node} has no tensor metadata associated with it! "
            f"Check that shape propagation has run."
        )

    return tensor_meta


@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    # [FUNCTION DOC] Computes element count of the node output from its tensor metadata, adds the
    # parameter element count of the submodule for call_module nodes, determines the per-element byte
    # size from the dtype (using a quantized empty tensor when the metadata says quantized), and returns
    # a size_bytes pair of output bytes and total bytes. Raises RuntimeError without tensor metadata.
    # KNOWN ISSUE: the docstring refers to node.dtype and node.shape, but the code reads tensor_meta.
    # Only the first tensor result is considered (tuple outputs are not supported).
    """Given a node with node.dtype and node.shape, return its total size and its output size.
    total_size = weights + bias + output_size
    """
    # Total num of elements
    total_num_of_elems = 0
    # For a module, consider all parameters
    if node.op == "call_module":
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        # Parameters are named tuples
        for _name, p in parameters:
            total_num_of_elems += p.numel()
    # Don't forget the output size
    # node.shape is the shape of this node's output
    tensor_meta = get_tensor_meta(node)
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    # Assume for now if it's quantized then it's qint8 or quint8
    if tensor_meta.is_quantized:
        size_per_elem_bytes = torch._empty_affine_quantized(
            [], dtype=tensor_meta.dtype
        ).element_size()
    else:
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)
