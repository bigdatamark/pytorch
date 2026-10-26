# [MODULE DOC] Serializes Inductor extern kernel nodes to a JSON string using the export serde schema.
import json

from torch._export.serde.schema import ExternKernelNode, ExternKernelNodes, Node
from torch._export.serde.serialize import _dataclass_to_dict, EnumEncoder
from torch._inductor.ir import ExternKernelNode as inductor_ExternKernelNode


def serialize_extern_kernel_node(
    extern_kernel_node: inductor_ExternKernelNode,
) -> ExternKernelNode:
    # [FUNCTION DOC] Converts an Inductor ExternKernelNode into the serde schema ExternKernelNode
    # (same name and node). Raises AssertionError if the wrapped node is not a serde Node.
    if not isinstance(extern_kernel_node.node, Node):
        raise AssertionError(
            f"expected node to be a Node, got {type(extern_kernel_node.node)}"
        )
    return ExternKernelNode(
        name=extern_kernel_node.name,
        node=extern_kernel_node.node,
    )


def extern_node_json_serializer(
    extern_kernel_nodes: list[inductor_ExternKernelNode],
) -> str:
    # [FUNCTION DOC] Serializes a list of Inductor extern kernel nodes into one JSON string by
    # converting each, wrapping them in the nodes container, turning it into plain mappings and
    # encoding with the enum-aware encoder. Returns the JSON text.
    serialized_nodes = ExternKernelNodes(
        nodes=[serialize_extern_kernel_node(node) for node in extern_kernel_nodes]
    )
    return json.dumps(_dataclass_to_dict(serialized_nodes), cls=EnumEncoder)
