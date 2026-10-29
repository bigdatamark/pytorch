# [MODULE DOC]
# Utility for propagating tensor properties (types) onto TorchScript graph
# inputs from example inputs. Not meant to be imported directly. No import-time
# side effects.
"""
Tools to help with tensor property propagation.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

from typing import Any

import torch
from torch import TensorType
from torch._C import Graph


def apply_input_props_using_example(graph: Graph, example_input: list[Any]) -> None:
    # [FUNCTION DOC] Sets tensor types on a graph's inputs from example values.
    # Drops a leading input named self of class type (method graphs), then
    # requires the remaining input count to equal len(example_input). None
    # examples are skipped; otherwise tensor-ness of example and input type must
    # agree, and tensor inputs get their type created from the example tensor.
    # Raises RuntimeError on count or type mismatch. Mutates the graph; returns None.
    """
    Applies properties for each tensor in the graph inputs
    using the example supplied.
    """
    graph_inputs = list(graph.inputs())
    if len(graph_inputs) == 0:
        return

    # Strip self args off for methods
    in_0 = graph_inputs[0]
    if isinstance(in_0.type(), torch._C.ClassType) and in_0.debugName() == "self":
        graph_inputs = graph_inputs[1:]

    if not len(graph_inputs) == len(example_input):
        raise RuntimeError(
            "Number of inputs in graph does not match number of inputs in the example"
        )

    for i, (graph_i, example_i) in enumerate(zip(graph_inputs, example_input)):
        if example_i is None:
            continue  # Skip the type check

        if isinstance(example_i, torch.Tensor) != isinstance(
            graph_i.type(), TensorType
        ):
            raise RuntimeError(
                f"Input {i} does not match type of example", graph_i, example_i
            )

        if isinstance(example_i, torch.Tensor):
            graph_i.setType(TensorType.create_from_tensor(example_i))  # type: ignore[arg-type]
