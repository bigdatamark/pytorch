# [MODULE DOC] Defines a single FX pass helper that propagates element type annotations from
# sequence-typed nodes (tuple, list, NamedTuple) onto the getitem nodes that index them.
# No import-time side effects.
import operator
import typing

import torch


def annotate_getitem_nodes(graph: torch.fx.Graph) -> None:
    # [FUNCTION DOC] Walks every node of the graph and, for each getitem call, sets the node's type
    # from the type of its sequence argument. Handles typing aliases with a name (Tuple/List), generic
    # aliases (tuple/list origin) and NamedTuple-like classes. Variadic tuples give their first element type;
    # fixed tuples use the type at the (integer) index; lists must have exactly one parameter type.
    # Sequence nodes without a type are skipped. Mutates node.type in place; returns nothing.
    # Raises AssertionError on an out-of-range tuple index or a list with other than one parameter.
    # KNOWN ISSUE: the tuple/list branches assume the index is an integer; a non-integer index node
    # (such as a Node or slice) would fail in the comparison or lookup.
    """
    Annotate the type of getitem nodes, inferred from the type of sequence node.
    If sequence node is not annotated with a type, do nothing.
    Currently support getitem nodes from tuple, list, and NamedTuple sequence node.

    This is helpful since annotations on local names within function are lost during FX transforms.
    Adding back known type annotation for getitem nodes to improve jit scriptability.

    Args:
        graph (:class:`torch.fx.Graph`): The graph to be annotated
    """
    for node in graph.nodes:
        if node.target is operator.getitem:
            sequence_node, index_node = node.args
            if not sequence_node.type:
                continue
            # container types
            if hasattr(sequence_node.type, "_name"):
                parameterized_types = typing.get_args(sequence_node.type)
                if sequence_node.type._name == "Tuple":
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        if len(parameterized_types) <= index_node:
                            raise AssertionError(
                                f"Index {index_node} out of range for parameterized_types "
                                f"(len={len(parameterized_types)})"
                            )
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type._name == "List":
                    if len(parameterized_types) != 1:
                        raise AssertionError(
                            f"Expected 1 parameterized type, got {len(parameterized_types)}"
                        )
                    node.type = parameterized_types[0]
            # Generic Alias Type
            elif hasattr(sequence_node.type, "__origin__"):
                parameterized_types = typing.get_args(sequence_node.type)
                if sequence_node.type.__origin__ is tuple:
                    if len(parameterized_types) == 2 and isinstance(
                        parameterized_types[1], type(...)
                    ):
                        node.type = parameterized_types[0]
                    else:
                        if len(parameterized_types) <= index_node:
                            raise AssertionError(
                                f"Index {index_node} out of range for parameterized_types "
                                f"(len={len(parameterized_types)})"
                            )
                        node_type = parameterized_types[index_node]
                        node.type = node_type
                elif sequence_node.type.__origin__ is list:
                    if len(parameterized_types) != 1:
                        raise AssertionError(
                            f"Expected 1 parameterized type, got {len(parameterized_types)}"
                        )
                    node.type = parameterized_types[0]
            # NamedTuple type
            elif hasattr(sequence_node.type, "__annotations__"):
                if sequence_node.type == torch.Tensor:
                    continue
                sequence_node_field_types = sequence_node.type.__annotations__
                field_name = sequence_node.type._fields[index_node]
                node.type = sequence_node_field_types[field_name]
