# [MODULE DOC] Small helpers for the auto chunker: tangent-node detection, fake-tensor
# lookup from fx node metadata, size/ndim/scalar queries, and accessors for the
# chunking metadata (ChunkingMeta) attached to fx nodes. No import-time side effects;
# core is imported lazily inside functions to avoid a circular import.
from collections.abc import Sequence
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_flatten

from .common import CantChunk, ChunkingMeta


def is_tangent_node(node: Node) -> bool:
    # [FUNCTION DOC] Returns True if the node is a placeholder whose name contains the substring tangent
    # (a backward-graph tangent input).
    return node.op == "placeholder" and "tangent" in node.name


def get_args_of_node_type(node: Node) -> Sequence[Node]:
    # [FUNCTION DOC] HELPER: Flattens args and kwargs of the node and returns the list of fx Node objects found in them.
    return [x for x in tree_flatten((node.args, node.kwargs))[0] if isinstance(x, Node)]


def use_tangent(node: Node) -> bool:
    # [FUNCTION DOC] Returns True if any Node argument (args or kwargs, flattened) of the node is a tangent placeholder.
    """
    Whether the fx node uses tangent input.
    """

    return any(
        is_tangent_node(arg)  # type: ignore[operator]
        for arg in get_args_of_node_type(node)
    )


def compute_tensor_size(*args: Any, count_bytes: bool = True, **kwargs: Any) -> int:
    # [FUNCTION DOC] Sums the element counts of all fake tensors found among the Node values in args and kwargs
    # (flattened). Multiplies by dtype itemsize when count_bytes is True, so the result is bytes; otherwise
    # elements. Non-node or non-tensor values are ignored. Returns an int.
    """
    Compute total tensor sizes from fx.Node in args & kwargs.
    """
    flat_args, _ = tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := get_fake_tensor_from_node_arg(arg)) is None:
            continue
        tot += fake_tensor.numel() * (fake_tensor.dtype.itemsize if count_bytes else 1)
    return tot


def get_fake_tensor_from_node_arg(
    node: torch.fx.node.Argument,
) -> torch.Tensor | None:
    # [FUNCTION DOC] Returns the fake tensor stored in the meta val entry of the given argument, or None when the
    # argument has no meta, no val entry, or val is not a tensor.
    if (
        not hasattr(node, "meta")
        or ("val" not in node.meta)  # type: ignore[union-attr]
        or not isinstance(node.meta["val"], torch.Tensor)  # type: ignore[union-attr]
    ):
        return None
    return node.meta["val"]  # type: ignore[union-attr]


def get_nodes_with_chunking_meta(graph: torch.fx.Graph) -> Sequence[Node]:
    # [FUNCTION DOC] Returns the list of graph nodes (in graph order) that carry chunking metadata.
    # Imports get_chunking_meta lazily from core.
    from .core import get_chunking_meta

    output = []
    for node in graph.nodes:
        if get_chunking_meta(node):
            output.append(node)
    return output


def format_node_with_chunking_meta(
    node: torch.fx.Node, include_args: bool = False
) -> None:
    # [FUNCTION DOC] HELPER: Debug printing. Prints the output shape (or ? if no fake tensor) and the formatted node,
    # then the chunking meta in green if present; if include_args is True also prints the chunking meta of
    # each Node argument. Side effect: writes to stdout. Returns None.
    """
    Print the node with chunking metadata for the current node if exists.

    If include_args is True, also print chunking metadata for Node arguments.
    """
    from torch._inductor.runtime.runtime_utils import green_text

    from .core import get_chunking_meta

    fake_tensor = get_fake_tensor_from_node_arg(node)
    shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
    print(f"  {shape} {node.format_node()}")

    if meta := get_chunking_meta(node):
        print(f"    {green_text(str(meta))}")

    if include_args:
        for arg in get_args_of_node_type(node):
            if arg_meta := get_chunking_meta(arg):
                print(f"    {arg}: {green_text(str(arg_meta))}")


def has_any_chunking_meta(*node_list: Node) -> bool:
    # [FUNCTION DOC] Returns True if at least one of the given nodes has chunking metadata.
    from .core import get_chunking_meta

    return any(get_chunking_meta(node) for node in node_list)


def get_first_chunking_meta(*node_list: Node) -> ChunkingMeta | None:
    # [FUNCTION DOC] Returns the chunking metadata of the first node in the given order that has one, else None.
    """
    Get the first non-none chunking metadata if there is any.
    """
    from .core import get_chunking_meta

    for node in node_list:
        if (meta := get_chunking_meta(node)) is not None:
            return meta

    return None


def get_scale_by_from_metas(*metas: ChunkingMeta) -> Node | None:
    # [FUNCTION DOC] Collects the scale_by field from each given ChunkingMeta (no deduplication).
    # Returns None if none has it, the single value if exactly one does.
    # Raises CantChunk if more than one meta has scale_by.
    """
    If there are multiple ChunkingMeta having the scale_by field,
    raise a CantChunk exception.

    If no ChunkingMeta has scale_by field, return None.
    Otherwise return the only scale_by field.
    """

    scale_by_list = []

    # don't do dedup on the scale_by field on purpose for this API
    for meta in metas:
        if meta.scale_by is not None:
            scale_by_list.append(meta.scale_by)

    if len(scale_by_list) > 1:
        raise CantChunk("Multiple scale_by")

    return scale_by_list[0] if len(scale_by_list) == 1 else None


def get_scale_by_from_node(node: Node) -> Node | None:
    # [FUNCTION DOC] Returns the scale_by node of the node's chunking metadata, or None if the node has no metadata
    # or no scale_by.
    from .core import get_chunking_meta

    meta = get_chunking_meta(node)
    return meta.scale_by if meta is not None else None


def get_node_is_scalar(nodes: Sequence[Node]) -> dict[Node, bool]:
    # [FUNCTION DOC] Returns a mapping from each given node to whether its fake tensor has exactly one element.
    # Raises AssertionError if any node has no fake tensor.
    """
    Returns a dict map a node to 'is_scalar'.
    """
    node_is_scalar = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        if ft is None:
            raise AssertionError(f"expected a fake tensor for node {node}, got None")
        node_is_scalar[node] = ft.numel() == 1
    return node_is_scalar


def get_node_ndim(nodes: Sequence[Node]) -> dict[Node, int]:
    # [FUNCTION DOC] Returns a mapping from each given node to the number of dimensions of its fake tensor.
    # Raises AssertionError if any node has no fake tensor.
    """
    Returns a dict map a node to 'ndim'.
    """
    node_ndim = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        if ft is None:
            raise AssertionError(f"expected a fake tensor for node {node}, got None")
        node_ndim[node] = ft.ndim
    return node_ndim


def is_chunked_by_dim(node: Node, dim: int) -> bool:
    # [FUNCTION DOC] Returns True if the node has chunking metadata and it indicates chunking along the given dim.
    from .core import get_chunking_meta

    meta = get_chunking_meta(node)
    return meta is not None and meta.chunked_by_dim(dim)


def tangent_has_chunking_meta(gm: GraphModule) -> bool:
    # [FUNCTION DOC] Returns True if any tangent placeholder in the graph module has chunking metadata.
    from .core import get_chunking_meta

    return any(
        is_tangent_node(node) and get_chunking_meta(node) is not None
        for node in gm.graph.find_nodes(op="placeholder", sort=False)
    )


def get_tangent_nodes(graph: Graph) -> Sequence[Node]:
    # [FUNCTION DOC] Returns the list of tangent placeholder nodes in the graph (unsorted search over placeholders).
    tangents = []
    for node in graph.find_nodes(op="placeholder", sort=False):
        if is_tangent_node(node):
            tangents.append(node)
    return tangents
