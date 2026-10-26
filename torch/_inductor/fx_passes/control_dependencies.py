# mypy: allow-untyped-defs
# [MODULE DOC] Defines the control_deps higher-order operator and a pass helper that wraps FX call nodes in it to impose explicit scheduling order.
# Contents: ControlDeps operator class and its singleton control_deps, fake/eager/autograd implementations, subgraph naming and creation helpers,
# and preserve_node_ordering. Import-time side effects: instantiates the operator, marks it as side-effecting so dead-code elimination keeps it,
# and registers its fake-tensor and dispatch implementations.
"""
Effect ordering pass for inductor.

This pass adds ordering dependencies to FX graphs using the control_deps HOP
for precise control over scheduling constraints. When you need exact ordering between
operations (e.g., collective_start -> mm -> wait), this pass wraps operations
with control_deps to make dependencies explicit.
"""

from operator import attrgetter
from typing import Any

import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


class ControlDeps(HigherOrderOperator):
    # [CLASS DOC] Higher-order operator control_deps(additional_deps, subgraph, *args, **kwargs). Runs the subgraph on args while recording that all
    # additional_deps must be computed first (ordering only). Stateless; named control_deps. Execution semantics equal just calling the subgraph.
    """
    Higher-order operator that enforces ordering by making dependencies explicit.

    Schema: control_deps(additional_deps, subgraph, *args, **kwargs) -> result
    where:
    - additional_deps: tuple of tensors that must be computed before this op
      (ordering-only, not a real data use)
    - subgraph: GraphModule containing the exact operation to execute
    - *args: pass-through arguments forwarded to the subgraph

    Semantics:
    - All tensors in additional_deps are computed before the subgraph executes.
    - Pass-through args (inputs returned unchanged by the subgraph) are
      versioned: future readers are ordered after all subgraph operations
      via a rename chain (OrderingOutput).  This ensures that consumers of
      a pass-through value cannot be scheduled before the subgraph's sync
      ops (e.g. wait_event) complete.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize the higher-order operator with the name control_deps.
        super().__init__("control_deps")

    def __call__(self, additional_deps, subgraph, *args, **kwargs):
        # [FUNCTION DOC] Validate and dispatch the operator. additional_deps must be a tuple or list and subgraph a GraphModule or callable; otherwise raises TypeError.
        # Then delegates to the base higher-order operator call, returning the subgraph result.
        """Call the operator with dependencies and subgraph.

        Args:
            additional_deps: Tuple of tensors that must be computed first
            subgraph: GraphModule containing the exact operation to execute
            *args: Arguments to pass to the subgraph
        """
        if not isinstance(additional_deps, (tuple, list)):
            raise TypeError(
                f"additional_deps must be tuple/list, got {type(additional_deps).__name__}"
            )
        if not (isinstance(subgraph, fx.GraphModule) or callable(subgraph)):
            raise TypeError(
                f"subgraph must be GraphModule or callable, got {type(subgraph).__name__}"
            )
        # pyrefly: ignore [missing-attribute]
        return super().__call__(additional_deps, subgraph, *args, **kwargs)


control_deps = ControlDeps()

# control_deps wraps side-effecting ops (e.g. record_event, wait_event)
# and must not be eliminated by DCE even when its outputs are unused.
from torch.fx.node import has_side_effect


has_side_effect(control_deps)


# Register fake implementation for tracing
@register_fake(control_deps)
def _(additional_deps, subgraph, *args, **kwargs):
    # [FUNCTION DOC] Fake-tensor implementation of control_deps: ignores additional_deps and returns the result of calling the subgraph on the arguments.
    """Fake tensor implementation - execute the subgraph."""
    return subgraph(*args, **kwargs)


# Register eager execution implementation
@control_deps.py_impl(DispatchKey.CompositeExplicitAutograd)
def control_deps_eager(additional_deps, subgraph, *args, **kwargs):
    # [FUNCTION DOC] Eager (CompositeExplicitAutograd) implementation: ignores additional_deps and runs the subgraph on the arguments.
    """Eager implementation - just execute the subgraph."""
    return subgraph(*args, **kwargs)


# Autograd impl needed because additional_deps tensors may have autograd state,
# causing dispatch through AutogradCUDA even in post-autograd graphs.
@control_deps.py_impl(DispatchKey.Autograd)
def control_deps_autograd(additional_deps, subgraph, *args, **kwargs):
    # [FUNCTION DOC] Autograd-key implementation: ignores additional_deps and runs the subgraph (needed because dependency tensors may carry autograd state).
    return subgraph(*args, **kwargs)


def get_subgraph_name(gm: fx.GraphModule, name):
    # [FUNCTION DOC] Return an unused attribute name on the module for a new subgraph: the prefix subgraph_ plus the given name,
    # with a numeric suffix appended (smallest free integer starting at 0) if that attribute already exists.
    name = f"subgraph_{name}"

    if not hasattr(gm, name):
        return name

    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1

    return f"{name}_{i}"


def _extract_unique_nodes(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[fx.Node], list[Any], Any]:
    # [FUNCTION DOC] HELPER: Flatten args and kwargs with pytree and collect unique fx Node instances in first-occurrence order.
    # Returns a triple: the unique node list, the full flattened list, and the pytree spec to rebuild the structure.
    """Extract unique fx.Node instances from args/kwargs using pytree.

    Args:
        args: The positional arguments (may contain nested structures with fx.Node)
        kwargs: The keyword arguments (may contain nested structures with fx.Node)

    Returns:
        - Ordered list of unique fx.Node instances (preserves first occurrence order)
        - Flattened list of all items from args/kwargs
        - The pytree spec for reconstructing the original structure
    """
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))
    unique_nodes: list[fx.Node] = []
    seen: OrderedSet[fx.Node] = OrderedSet()
    for item in flat_args_kwargs:
        if isinstance(item, fx.Node) and item not in seen:
            unique_nodes.append(item)
            seen.add(item)
    return unique_nodes, flat_args_kwargs, spec


def preserve_node_ordering(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    # [FUNCTION DOC] For each dependent node in the mapping (node to ordered set of dependency nodes), replace it with a control_deps call node.
    # Builds a subgraph module reproducing the original operation, stores it as a new attribute of the owning module, inserts a get_attr for it
    # and a control_deps call taking the dependencies (remapped to already-replaced nodes) and the original node arguments.
    # Copies metadata (dropping eager_input_vals), redirects all users, erases the original and gives the new node the original name.
    # Does nothing for an empty mapping. Raises AssertionError if a node is not call_function or the graph has no owning module. Mutates the graph. The verbose flag is unused.
    """
    Preserve node ordering using control_deps HOP with subgraph.

    This function wraps operations with control_deps that:
    1. Makes additional dependencies explicit (first argument)
    2. Creates a subgraph internally to preserve the exact original operation
    3. Preserves the original node names

    Args:
        graph: The FX graph to modify
        additional_deps_map: Mapping from dependent nodes to their dependencies
        verbose: If True, print debug information
    """
    if not additional_deps_map:
        return

    # Track replacements so we can update dependencies
    replacements: dict[fx.Node, fx.Node] = {}

    # Process each node that needs additional dependencies
    for dependent_node, dep_nodes in additional_deps_map.items():
        if dependent_node.op != "call_function":
            raise AssertionError(dependent_node.op)

        original_name = dependent_node.name
        original_args = dependent_node.args
        original_kwargs = dependent_node.kwargs
        original_meta = dependent_node.meta.copy()

        updated_dep_nodes = [replacements.get(dep, dep) for dep in dep_nodes]

        # Create a subgraph that preserves the exact original operation
        subgraph_module = _create_subgraph_for_node(graph, dependent_node)

        owning_mod = graph.owning_module
        if owning_mod is None:
            raise AssertionError("expected graph to have an owning_module")
        subgraph_attr_name = get_subgraph_name(owning_mod, original_name)
        setattr(graph.owning_module, subgraph_attr_name, subgraph_module)

        # Create control_deps call with:
        # 1. Additional dependencies as first arg (explicit)
        # 2. Subgraph via get_attr (like b2b gemm pass)
        # 3. Original arguments (only fx.Node args and kwargs are passed)
        with graph.inserting_before(dependent_node):
            # Create get_attr node for the subgraph
            get_subgraph = graph.get_attr(subgraph_attr_name)

            # Extract unique nodes from nested args/kwargs
            node_args, _, _ = _extract_unique_nodes(original_args, original_kwargs)

            # Create with temporary name first
            ordered_node = graph.call_function(
                control_deps,
                args=(
                    tuple(updated_dep_nodes),  # additional_deps
                    get_subgraph,  # subgraph via get_attr (like b2b gemm)
                    *node_args,  # original node arguments (from both args and kwargs)
                ),
                kwargs={},
                name=f"__temp_{original_name}",  # Temporary name to avoid conflict
            )

        # Copy metadata from original node
        ordered_node.meta = original_meta
        # this will be constrained on the target node in subgraph if it exists
        ordered_node.meta.pop("eager_input_vals", None)

        # Replace all uses of the original node with the ordered version
        dependent_node.replace_all_uses_with(ordered_node)

        # Remove the original node from the graph
        graph.erase_node(dependent_node)

        # Now rename the ordered node to the original name
        ordered_node.name = original_name  # PRESERVE ORIGINAL NAME

        # Track the replacement for future dependencies
        replacements[dependent_node] = ordered_node


def _create_subgraph_for_node(
    graph: fx.Graph, node: fx.Node, additional_deps=None
) -> fx.GraphModule:
    # [FUNCTION DOC] HELPER: Build a GraphModule that recreates a node's operation: one placeholder per unique fx Node argument (copying value metadata, or
    # the attribute value for get_attr nodes), optional extra placeholders for additional_deps, a call of the same target with rebuilt args and kwargs.
    # Output is the result alone, or the result followed by the passed-through dep placeholders, with value metadata set.
    # Raises AssertionError if the graph lacks an owning module or the target is not callable. Returns a lazy graph module.
    """
    Create a subgraph that exactly recreates a node's operation optionally passing through additional dependencies.

    The subgraph takes only the fx.Node arguments and recreates the operation
    with the exact target, args structure, and kwargs.

    Args:
        graph: The parent graph
        node: The node to wrap in a subgraph
        additional_deps: Additional dependencies to pass through the subgraph

    Returns:
        A GraphModule containing the subgraph
    """
    # Get the owning module
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("graph.owning_module must not be None")

    # Create a new graph for the subgraph
    subgraph = fx.Graph(owning_module)

    # Extract unique nodes and get flattened structure + spec
    unique_nodes, flat_args_kwargs, spec = _extract_unique_nodes(node.args, node.kwargs)

    # Create placeholders for each unique node
    node_to_placeholder: dict[fx.Node, fx.Node] = {}
    for idx, orig_node in enumerate(unique_nodes):
        placeholder = subgraph.placeholder(f"arg_{idx}")
        if "val" in orig_node.meta:
            placeholder.meta.update(orig_node.meta)
        elif orig_node.op == "get_attr" and isinstance(orig_node.target, str):
            placeholder.meta["val"] = attrgetter(orig_node.target)(owning_module)
        node_to_placeholder[orig_node] = placeholder

    # Replace fx.Node instances with their placeholders
    def replace_nodes(item: Any) -> Any:
        # [FUNCTION DOC] HELPER: Map an fx Node to its subgraph placeholder; return any other item unchanged.
        if isinstance(item, fx.Node):
            return node_to_placeholder[item]
        return item

    additional_deps_placeholders = []
    for idx, dep in enumerate(additional_deps or ()):
        placeholder = subgraph.placeholder(f"dep_{idx}")
        if "val" in dep.meta:
            placeholder.meta.update(dep.meta)
        additional_deps_placeholders.append(placeholder)

    new_flat = [replace_nodes(item) for item in flat_args_kwargs]
    new_args, new_kwargs = pytree.tree_unflatten(new_flat, spec)

    # Recreate the exact original operation in the subgraph
    if not callable(node.target):
        raise AssertionError(f"expected node.target to be callable, got {node.target}")
    result = subgraph.call_function(
        node.target,
        tuple(new_args),
        new_kwargs,  # type: ignore[arg-type]
    )

    # Copy metadata from the original node
    result.meta.update(node.meta)

    if additional_deps_placeholders:
        outputs = tuple([result] + additional_deps_placeholders)
        out = subgraph.output(outputs)
        out.meta["val"] = tuple(output.meta.get("val") for output in outputs)
    else:
        out = subgraph.output(result)
        if "val" in result.meta:
            out.meta["val"] = result.meta["val"]

    return _LazyGraphModule(owning_module, subgraph)
