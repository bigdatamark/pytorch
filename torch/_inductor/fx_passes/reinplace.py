# mypy: allow-untyped-defs
# [MODULE DOC] Inductor FX pass that turns functional ops into in-place variants when it is safe: index_put, generalized
# view-scatter ops, philox RNG, collectives, foreach ops, auto_functionalized (v1/v2), with_effects wrappers and
# functional triton kernel wrappers. Also canonicalizes view_scatter ops into a generalized scatter and decomposes it back.
# Main entry point: reinplace_inplaceable_ops. Import time builds the tables inplaceable_ops, inplaceable_foreach_ops,
# inplaceable_triton_ops and META_ONLY_OPS (collective ops only if torch.ops._c10d_functional exists).
import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.fx.node
from torch._C._dynamo.guards import compute_overlapping_tensors
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import ReinplaceCounters, ReInplaceTrigger
from torch._guards import detect_fake_mode
from torch._higher_order_ops.triton_kernel_wrap import (
    kernel_side_table,
    triton_kernel_wrapper_functional,
)
from torch._inductor import config, inductor_prims
from torch._inductor.fx_utils import get_node_storage, is_node_realized
from torch._inductor.lowering import (
    inplaceable_foreach_ops as inplaceable_foreach_ops_lowerings,
)
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    GuardOnDataDependentSymNode,
    statically_known_true,
    sym_eq,
)
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.reinplace import _is_view_op
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass(frozen=True)
class InplaceableOp:
    # [CLASS DOC] Frozen record pairing an out-of-place op's in-place replacement with the argument index (or tuple of indices)
    # it mutates. extra_check is a predicate on the node that must also hold before reinplacing (default: always true).
    inplace_op: Callable[..., Any]
    mutated_arg: int | tuple[int, ...]  # Single index or tuple of indices
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True

    @property
    def mutated_args(self) -> tuple[int, ...]:
        # [FUNCTION DOC] Returns mutated_arg normalized to a tuple of indices (wraps a single int).
        """Return mutated_arg as a tuple for uniform handling."""
        if isinstance(self.mutated_arg, int):
            return (self.mutated_arg,)
        return self.mutated_arg


_SCATTER_OP_TO_VIEW = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}
_VIEW_OP_TO_SCATTER = {v: k for k, v in _SCATTER_OP_TO_VIEW.items()}


def graph_call_function(graph: torch.fx.Graph, fn, *args, **kwargs):
    # [FUNCTION DOC] Appends a call_function node to graph while computing its fake output value: argument nodes are replaced
    # by their meta val, fn is run under V.fake_mode, and the result is stored in node.meta val.
    # If the fake mode has a shape env and the result has unbacked symbols, stores unbacked_bindings in meta. Returns the node.
    fake_args, fake_kwargs = pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
        (args, kwargs),
    )
    with V.fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)

    node = graph.call_function(fn, args, kwargs)

    node.meta["val"] = fake_result
    if (V.fake_mode.shape_env) and (
        symbol_to_path := compute_unbacked_bindings(V.fake_mode.shape_env, fake_result)
    ):
        node.meta["unbacked_bindings"] = symbol_to_path

    return node


@dataclass
class ViewOp:
    # [CLASS DOC] Plain record of one view operation: target OpOverload plus its extra args and kwargs (excluding the viewed tensor).
    target: torch._ops.OpOverload
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _inplace_generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Reference semantics of the in-place generalized scatter: applies each ViewOp in sequence to inp
    # (arguments are fake values from node metadata, with fresh unbacked symbols ignored), then copies src into the final view.
    # Raises RuntimeError with a shape message if src cannot broadcast into the view. Returns inp (mutated).
    tmp = inp
    for view in view_ops:
        fake_args, fake_kwargs = pytree.tree_map(
            lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
            (view.args, view.kwargs),
        )
        # slice and select can allocate new unbacked symints, but those won't be reflected
        # in the output of this function, hence shall be ignored.
        fake_mode = detect_fake_mode(fake_args)
        with (
            fake_mode.shape_env.ignore_fresh_unbacked_symbols()
            if fake_mode and fake_mode.shape_env
            else nullcontext()
        ):
            tmp = view.target(tmp, *fake_args, **fake_kwargs)
    try:
        tmp.copy_(src)
    except RuntimeError as e:
        raise RuntimeError(
            f"shape error in scatter op, can not broadcast {src.shape} to {tmp.shape}"
        ) from e
    return inp


def _generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: list[ViewOp]
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Functional generalized scatter: clones inp, then applies the in-place version. Used as a graph target; returns the updated clone.
    out = inp.clone()
    return _inplace_generalized_scatter(out, src, view_ops)


def _decompose_scatter_functional_helper(
    graph: torch.fx.Graph,
    inp: torch.Tensor,
    src: torch.Tensor,
    view_ops: list[ViewOp],
) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Recursively emits graph nodes for a functional scatter: for a chain of view ops, creates the first view of inp,
    # recursively scatters src into that view, then emits the matching view_scatter op for the first view with the result.
    # Returns the final node (the updated inp).
    view_op, view_ops_tail = view_ops[0], view_ops[1:]

    if view_ops_tail:
        view = graph_call_function(
            graph, view_op.target, inp, *view_op.args, **view_op.kwargs
        )
        src = _decompose_scatter_functional_helper(graph, view, src, view_ops[1:])  # type: ignore[assignment]

    return graph_call_function(
        graph,
        _VIEW_OP_TO_SCATTER[view_op.target],
        inp,
        src,
        *view_op.args,
        **view_op.kwargs,
    )


def _decompose_scatter_functional(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Expands a _generalized_scatter node into nested view and view_scatter ops (see docstring example).
    # Raises AssertionError if node.target is not _generalized_scatter. Returns the new node standing for the updated input.
    """Decompose _generalized_scatter to a sequence of view_scatter operations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    view = aten.slice(inp, 0, 0, 10)
    view_updated = aten.slice_scatter(view, src, 1, 10, -10)
    inp_updated = aten.slice_scatter(inp, view_updated, 0, 0, 10)
    """
    if node.target is not _generalized_scatter:
        raise AssertionError(
            f"expected node.target to be _generalized_scatter, got {node.target}"
        )
    return _decompose_scatter_functional_helper(graph, *node.args)  # type: ignore[arg-type]


def _decompose_scatter_mutating(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Expands a generalized scatter node into mutation: clone of inp (only for the functional variant), a chain of views,
    # and a copy_ of src into the last view. Raises AssertionError for unexpected target or if the node has kwargs.
    # Returns the (cloned or original) input node which holds the updated value.
    """Decompose _generalized_scatter using mutations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    inp_updated = aten.clone(inp)
    slice1 = aten.slice(inp_updated, 0, 0, 10)
    slice2 = aten.slice(slice1, 1, 10, -10)
    slice2.copy_(src)

    """
    if node.target not in (_generalized_scatter, _inplace_generalized_scatter):
        raise AssertionError(f"unexpected node.target: {node.target}")
    inp, src, view_ops = node.args
    if node.kwargs:
        raise AssertionError(f"expected no kwargs, got {node.kwargs}")

    if node.target is _generalized_scatter:
        inp = graph_call_function(graph, aten.clone, inp)

    tmp = inp
    for view in view_ops:  # type: ignore[union-attr]
        tmp = graph_call_function(graph, view.target, tmp, *view.args, **view.kwargs)  # type: ignore[union-attr]

    graph_call_function(graph, aten.copy_.default, tmp, src)
    return inp  # type: ignore[return-value]


# View ops whose view_scatter op is lowered into mutations anyway,
# so is never a pessimisation to decompose.
_ALWAYS_MUTATING_SCATTER_OPS = OrderedSet(
    [
        aten.as_strided.default,
        aten.diagonal.default,
    ]
)


def scatter_always_uses_mutation(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] True if any view op in the scatter node's view list is an OpOverload in _ALWAYS_MUTATING_SCATTER_OPS (as_strided, diagonal),
    # whose scatter lowering mutates anyway. Reads the view list from the node's third arg.
    _, _, view_ops = node.args
    view_ops = cast(Sequence[torch.fx.node.Argument], view_ops)
    return any(
        target in _ALWAYS_MUTATING_SCATTER_OPS
        for view in view_ops
        if isinstance(target := getattr(view, "target", None), torch._ops.OpOverload)
    )


def should_reinplace_scatter(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] Decides whether a generalized scatter should be reinplaced (mutating) rather than kept functional.
    # True if it always mutates, if both input and output are realized, or if the input is a placeholder/get_attr that is copied back
    # by a user copy_. Otherwise False, assuming fusion makes the functional form cheaper.
    """Choose between mutating and functional scatter decompositions

    Reinplacing view scatter ops can be pessimising as it blocks fusion with the
    input or output tensor computations. However, it is still profitable if the
    input and output would have been realized anyway.

    """
    inp, _src, _view_ops = node.args

    # Mutating scatter ops unconditionally realize input and output
    if scatter_always_uses_mutation(node):
        return True

    if is_node_realized(inp) and is_node_realized(node):  # type: ignore[arg-type]
        return True

    # If the output is copied back into the input, this forces both to be
    # realized as the output is a user of the input
    if inp.op in ("placeholder", "get_attr") and any(  # type: ignore[union-attr]
        user.target is aten.copy_.default and user.args[0] is inp for user in node.users
    ):
        return True

    # Otherwise, assume fusions will make functional variants profitable
    return False


def decompose_generalized_scatter(graph: torch.fx.Graph) -> None:
    # [FUNCTION DOC] Replaces every remaining _generalized_scatter and _inplace_generalized_scatter node with ordinary aten ops, using the
    # mutating decomposition for in-place or always-mutating scatters and the functional one otherwise.
    # Rewires all uses to the new node and erases the old one. Mutates graph.
    """Replace _generalized_scatter with normal aten ops"""
    for node in itertools.chain(
        graph.find_nodes(op="call_function", target=_generalized_scatter),
        graph.find_nodes(op="call_function", target=_inplace_generalized_scatter),
    ):
        use_mutation = (
            node.target is _inplace_generalized_scatter
            or scatter_always_uses_mutation(node)
        )

        with graph.inserting_before(node):
            if use_mutation:
                new_node = _decompose_scatter_mutating(graph, node)
            else:
                new_node = _decompose_scatter_functional(graph, node)

        node.replace_all_uses_with(new_node)
        graph.erase_node(node)


def canonicalize_view_scatter_ops(graph: torch.fx.Graph) -> None:
    # [FUNCTION DOC] Rewrites view_scatter ops (diagonal/select/slice/as_strided scatter) into _generalized_scatter nodes and fuses
    # a scatter whose source is a generalized scatter of a nested view of the same base into one scatter with a combined view list,
    # re-creating the intermediate view from the fused result if still used. Tracks per-node view base and view-op chains while
    # walking the graph. Mutates graph; returns None. Raises AssertionError on malformed scatter nodes.
    """
    This canonicalizes view scatter ops into a generalized form, defined as:
      def scatter(inp, src, views):
        tmp = inp.clone()
        for view in views:
          tmp = view(tmp)
        tmp.copy_(src)

    We also fuse consecutive view scatter ops of the form
        a = scatter(view2(self), src, [view1])
        b = scatter(self, a, [view2])
    which can be rewritten as
        b = scatter(self, src, [view2, view1])
        a = view2(b)

    This is both more efficient as we only do a single scatter, and also
    easier to reinplace since there is only one use of `self`
    """

    node_to_view_base: dict[torch.fx.Node, torch.fx.Node] = {}
    node_to_view_op: dict[torch.fx.Node, list[ViewOp]] = defaultdict(list)

    def handle_views(node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: Records for a view node its root base and the accumulated list of ViewOps from that base.
        inp = node.args[0]
        node_to_view_base[node] = node_to_view_base.get(inp, inp)  # type: ignore[arg-type, assignment]
        node_to_view_op[node] = [
            *node_to_view_op[inp],  # type: ignore[index]
            ViewOp(
                node.target,  # type: ignore[arg-type]
                args=node.args[1:],
                kwargs=node.kwargs,
            ),
        ]

    def handle_view_scatter(node: torch.fx.Node):
        # [FUNCTION DOC] HELPER: Converts one view scatter node into a generalized scatter node, fusing with a source generalized scatter when can_fuse holds.
        # Replaces uses and erases the old node (and the fused source node); recreates the source view for remaining users.
        # Raises AssertionError if args are too few or the target is not an OpOverload.
        if len(node.args) < 2:
            raise AssertionError(f"expected at least 2 args, got {len(node.args)}")
        inp, src = node.args[:2]

        if not isinstance(node.target, torch._ops.OpOverload):
            raise AssertionError(
                f"expected node.target to be an OpOverload, got {type(node.target)}"
            )
        scatter_view_op = ViewOp(
            _SCATTER_OP_TO_VIEW[node.target],
            args=node.args[2:],
            kwargs=node.kwargs,
        )

        def can_fuse():
            # [FUNCTION DOC] HELPER: True if src is a generalized scatter on the same view base whose view chain equals the input's chain plus this scatter's view op.
            if src.target is not _generalized_scatter:  # type: ignore[union-attr]
                return False
            src_inp, _src_src, _src_scatter_view_op = src.args  # type: ignore[union-attr]

            inp_base = node_to_view_base.get(inp, inp)  # type: ignore[arg-type]
            src_base = node_to_view_base.get(src_inp, src_inp)  # type: ignore[arg-type]
            return inp_base is src_base and node_to_view_op[src_inp] == [  # type: ignore[index]
                *node_to_view_op[inp],  # type: ignore[index]
                scatter_view_op,
            ]

        if not can_fuse():
            with graph.inserting_before(node):
                new_node = graph_call_function(
                    graph,
                    _generalized_scatter,
                    inp,
                    src,
                    [scatter_view_op],
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            return

        _src_inp, src_src, src_scatter_view_op = src.args  # type: ignore[union-attr]
        with graph.inserting_before(src):  # type: ignore[arg-type]
            new_node = graph_call_function(
                graph,
                _generalized_scatter,
                inp,
                src_src,
                [scatter_view_op, *src_scatter_view_op],  # type: ignore[misc]
            )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

            if src.users:  # type: ignore[union-attr]
                new_src = graph_call_function(
                    graph,
                    _SCATTER_OP_TO_VIEW[node.target],
                    new_node,
                    *node.args[2:],
                    **node.kwargs,
                )

                handle_views(new_src)
                src.replace_all_uses_with(new_src)  # type: ignore[union-attr]

            graph.erase_node(src)  # type: ignore[arg-type]

    for node in graph.nodes:
        if _is_view_op(node.target):
            handle_views(node)
        elif node.target in _SCATTER_OP_TO_VIEW:
            handle_view_scatter(node)


inplaceable_ops: dict[Callable[..., Any], InplaceableOp] = {
    aten.index_put.default: InplaceableOp(aten.index_put_.default, 0),
    aten._unsafe_index_put.default: InplaceableOp(inductor_prims._unsafe_index_put_, 0),
    _generalized_scatter: InplaceableOp(
        _inplace_generalized_scatter,
        0,
        extra_check=should_reinplace_scatter,
    ),
    # Stateless Philox RNG: reinplace the functionalized clone onto the dead
    # output buffer, so out-of-place uniform()/normal() don't pay an extra copy.
    aten._philox_uniform.default: InplaceableOp(aten._philox_uniform_.default, 0),
    aten._philox_normal.default: InplaceableOp(aten._philox_normal_.default, 0),
}

try:
    c10d_functional = torch.ops._c10d_functional
    inplaceable_collective_ops: dict[Callable[..., Any], InplaceableOp] = {
        c10d_functional.all_reduce.default: InplaceableOp(
            c10d_functional.all_reduce_.default, 0
        ),
        c10d_functional.all_reduce_coalesced.default: InplaceableOp(
            c10d_functional.all_reduce_coalesced_.default, 0
        ),
    }
    inplaceable_ops.update(inplaceable_collective_ops)
except AttributeError:
    # _c10d_functional ops are only available when torch
    # is built with USE_DISTRIBUTED=1.
    pass

inplaceable_foreach_ops: dict[torch._ops.OpOverload, InplaceableOp] = {}
for outplace_op, inplace_op in inplaceable_foreach_ops_lowerings.items():
    inplaceable_foreach_ops[outplace_op] = InplaceableOp(inplace_op, 0)


inplaceable_triton_ops = OrderedSet([triton_kernel_wrapper_functional])


# Operators that don't depend on the tensor data
META_ONLY_OPS = OrderedSet(
    [
        aten.sym_size.int,
        aten.sym_stride.int,
        aten.sym_numel.default,
        aten.sym_storage_offset.default,
    ]
)


def _get_view_base(node: torch.fx.Node) -> torch.fx.Node:
    # [FUNCTION DOC] HELPER: Follows first arguments through view ops and returns the root non-view node (stops if the first arg is not a Node).
    while _is_view_op(node.target) and node.args:
        base = node.args[0]
        if not isinstance(base, torch.fx.Node):
            break
        node = base
    return node


def _same_tensor_metadata(lhs: torch.fx.Node, rhs: torch.fx.Node) -> bool:
    # [FUNCTION DOC] HELPER: True if both nodes have tensor fake values with statically known equal sizes, strides and storage offset; False otherwise.
    lhs_val = lhs.meta.get("val")
    rhs_val = rhs.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return False

    def same_value(lhs_value, rhs_value) -> bool:
        # [FUNCTION DOC] HELPER: True if the two possibly symbolic values are statically known to be equal.
        return statically_known_true(sym_eq(lhs_value, rhs_value))

    def same_sequence(lhs_values, rhs_values) -> bool:
        # [FUNCTION DOC] HELPER: True if two sequences have equal length and all elements pairwise statically equal.
        return len(lhs_values) == len(rhs_values) and all(
            same_value(lhs_value, rhs_value)
            for lhs_value, rhs_value in zip(lhs_values, rhs_values)
        )

    return (
        same_sequence(lhs_val.size(), rhs_val.size())
        and same_sequence(lhs_val.stride(), rhs_val.stride())
        and same_value(lhs_val.storage_offset(), rhs_val.storage_offset())
    )


def _is_layout_preserving_view_copy_back(
    dst: torch.fx.Node,
    src: torch.fx.Node,
    mutated_arg: torch.fx.Node,
    src_base: torch.fx.Node,
) -> bool:
    # [FUNCTION DOC] HELPER: True if the base of the written view matches the mutated arg in layout and the source matches the destination in layout,
    # so that copying the view back into a graph input is equivalent to mutating that input in place.
    return _same_tensor_metadata(src_base, mutated_arg) and _same_tensor_metadata(
        src, dst
    )


def _is_control_deps_ordering_only_use(
    user: torch.fx.Node, view: torch.fx.Node
) -> bool:
    # [FUNCTION DOC] HELPER: True if user is a control_deps higher-order call in which view appears only in the ordering-only deps tuple (first arg)
    # and not among the passed-through arguments, i.e. not a real data use.
    """Check if view appears only in control_deps' ordering-only additional_deps tuple.

    control_deps schema (control_dependencies.py): control_deps(additional_deps, subgraph, *args)
    kwargs is always empty -- preserve_node_ordering threads all nodes positionally.
    """
    if not (
        user.op == "call_function"
        and isinstance(user.target, torch._ops.HigherOrderOperator)
        and user.target._name == "control_deps"
    ):
        return False
    additional_deps = cast(tuple[torch.fx.Node, ...], user.args[0])
    pass_through = cast(tuple[torch.fx.Node, ...], user.args[2:])
    return view in additional_deps and view not in pass_through


def reinplace_inplaceable_ops_core(graph: torch.fx.Graph) -> None:
    # [FUNCTION DOC] Core reinplacing pass over graph (mutated in place). First scans for copy_ nodes writing into placeholders/get_attr and builds
    # node order, storage-to-nodes maps and copy-back tables (direct and via views). Then visits each node: ops in inplaceable_ops are retargeted to
    # their in-place op when all mutated args can be reinplaced and extra_check holds; auto_functionalized (v1/v2) and triton wrappers get a reduced
    # clone list in metadata or kwargs; with_effects wrappers get the inner op swapped and getitem uses redirected; foreach ops are retargeted
    # when every arg is copied back. Finally redundant copy_ and getitem nodes are replaced and erased. A tensor is reinplaceable only if no later use
    # of any aliasing view observes it (inputs also need a copy_ epilogue). Raises AssertionError on inconsistent inputs or unknown triton kernel type.
    """
    Reinplaces in-placeable operations.
    If there are no uses of a view of the mutated arg after the current node,
    it is possible to inplace the op.
    This above algorithm could be justified by observing side effects. While
    we traverse the graph in forwards direction, only latter nodes could view
    side effects of the current node. If the current node is not used later as
    well as no view of this node is used later in the graph, then it is safe to
    inplace as there would be no way to observe the side effects.
    This condition is slightly different for graph inputs where they can only
    be inplaced if the above condition is true and there's a copy_ in the
    epilogue that signals that the caller wants to observe the mutation.

    Unlike JIT Inductor, AOTInductor currently unlifts weights and buffers from
    input args, so instead of checking mutation on placeholder, AOTInductor
    checks mutation on get_attr. This is subject to change in future.
    """

    copy_args_to_copy_nodes = {}
    # maps argument to the first copy_ node that mutates it.
    copy_nodes = {}
    # maps (view_arg, inplaceable_op_node) to copy_ nodes that copy a view of
    # the op result back into the graph input that view_arg aliases.
    copy_args_to_copy_nodes_via_views = {}
    mutated_inputs = OrderedSet[Any]()
    storage_to_nodes = defaultdict(list)
    node_order: dict[Any, int] = {}
    for i, node in enumerate(reversed(graph.nodes)):
        node_order[node] = len(graph.nodes) - i - 1
        storage_to_nodes[get_node_storage(node)].append(node)
        if node.target is aten.copy_.default and node.args[0].op in (
            "placeholder",
            "get_attr",
        ):
            dst = node.args[0]
            src = node.args[1]
            # If the target is a getitem and it indexes a possible clone,
            # then skip over it
            if src.target is operator.getitem and (
                (
                    src.args[0].target == triton_kernel_wrapper_functional
                    and src.args[0].kwargs["kwargs"][src.args[1]] == node.args[0]
                )
                or (src.args[0].target in inplaceable_foreach_ops)
                or (src.args[0].target is torch.ops.higher_order.auto_functionalized)
            ):
                src = src.args[0]

            copy_args_to_copy_nodes[(dst, src)] = node
            copy_nodes[dst] = node

            mutated_inputs.add(node.args[0])

            src_base = _get_view_base(src)
            if src_base is not src and isinstance(src_base.target, Callable):
                inplaceable_op = inplaceable_ops.get(src_base.target)
                if inplaceable_op is not None:
                    for mutated_arg_idx in inplaceable_op.mutated_args:
                        mutated_arg = src_base.args[mutated_arg_idx]
                        if (
                            isinstance(mutated_arg, torch.fx.Node)
                            and _get_view_base(mutated_arg) is dst
                            and _is_layout_preserving_view_copy_back(
                                dst, src, mutated_arg, src_base
                            )
                        ):
                            copy_args_to_copy_nodes_via_views[
                                (mutated_arg, src_base)
                            ] = node

    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node, mutated_arg):
        # [FUNCTION DOC] HELPER: True if some user of the given aliasing views occurs after node and could observe the mutation. Ignores users at or
        # after copy_node (the epilogue), meta-only users, copy_ into mutated_arg itself, and control_deps ordering-only uses.
        node_loc = node_order[node]
        copy_node_loc = node_order[copy_node] if copy_node is not None else None

        def is_meta_only_user(node):
            # [FUNCTION DOC] HELPER: True if node is a size/stride/numel/storage-offset op, or a view whose users are all meta-only.
            if _is_view_op(node.target):
                return all(is_meta_only_user(u) for u in node.users)
            return node.target in META_ONLY_OPS

        for view in shared_view_nodes:
            for user in view.users:
                user_loc = node_order[user]
                # Skip all users before node
                if user_loc <= node_loc:
                    continue
                # Ignore uses after the copy_ epilogue node, where the input
                # has already been mutated anyway
                if copy_node_loc is not None and copy_node_loc <= user_loc:
                    continue
                # Reinplacing does not change shape metadata
                if is_meta_only_user(user):
                    continue
                # If our graph looks like:
                # foo(mutated_arg)
                # mutated_arg.copy_(other)
                # then it's safe for us to reinplace foo because mutated_arg
                # will get overwritten anyways.
                if (
                    user.target is torch.ops.aten.copy_.default
                    and mutated_arg is user.args[0]
                ):
                    continue
                # control_deps args[0] is an ordering-only deps tuple;
                # appearing there does not constitute a real data use.
                if _is_control_deps_ordering_only_use(user, view):
                    continue
                return True
        return False

    def can_inplace(node, mutated_arg):
        # [FUNCTION DOC] Decides if mutated_arg (a node, or list/tuple of nodes) can be mutated in place by node. Lists require distinct storages and all elements ok.
        # Rejects args without storage or with internal overlap. For graph inputs requires a copy_ back and no later view uses; for views of inputs requires
        # a layout-preserving copy back of this op's result; otherwise requires no later use of overlapping views. Returns bool.
        # ls should be a list of tensors that all shares the same storage.
        def _overlap(ls) -> bool:
            # [FUNCTION DOC] HELPER: True if the given tensors overlap in memory; assumes overlap if a data-dependent guard error occurs.
            try:
                return len(compute_overlapping_tensors(ls)) != 0
            except GuardOnDataDependentSymNode:
                # If we fail with data dependent error we assume they all overlap.
                return True

        if isinstance(mutated_arg, (list, tuple)):
            # TODO Using _overlap here causes a several issues.
            unique_storages = OrderedSet(get_node_storage(arg) for arg in mutated_arg)
            if len(unique_storages) != len(mutated_arg):
                # At least two Tensors in mutated_arg alias each other, so we can't reinplace it.
                # We can probably do better (that is, reinplace one of them and clone the other)
                # but that requires more work and mutable List[Tensor] are not that common.
                return False
            return all(can_inplace(node, arg) for arg in mutated_arg)

        if get_node_storage(mutated_arg) is None:
            return False

        if torch._debug_has_internal_overlap(mutated_arg.meta["val"]) == 1:
            return False

        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]

        # Only keep tensor that might overlap with mutated_arg.
        shared_view_nodes = [
            v
            for v in shared_view_nodes
            if _overlap([mutated_arg.meta["val"], v.meta["val"]])
        ]

        if mutated_arg.op in ("placeholder", "get_attr"):
            # Get the first copy_ node that mutates the mutated_arg.
            copy_node = copy_nodes.get(mutated_arg)
            if copy_node is None:
                # There is no copy_ back to the candidate mutated_arg (which is a graph input).
                # Therefore the semantics of the program are that it does not mutate
                # mutated_arg, so we cannot re-inplace it.
                return False
            if any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=copy_node, mutated_arg=mutated_arg
            ):
                return False

            return True
        elif any(view.op in ("placeholder", "get_attr") for view in shared_view_nodes):
            # If mutated_arg is a view of a graph input, we can only reinplace
            # when a later copy_ writes this op's result back to that input.
            copy_node = copy_args_to_copy_nodes_via_views.get((mutated_arg, node))
            if copy_node is None:
                return False

            mutated_arg_base = copy_node.args[0]
            copy_src = copy_node.args[1]
            if not (
                isinstance(mutated_arg_base, torch.fx.Node)
                and isinstance(copy_src, torch.fx.Node)
                and _is_layout_preserving_view_copy_back(
                    mutated_arg_base, copy_src, mutated_arg, node
                )
            ):
                return False

            if any_use_of_views_after_node(
                node,
                shared_view_nodes,
                copy_node=copy_node,
                mutated_arg=mutated_arg_base,
            ):
                return False
            return True
        else:
            return not any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=None, mutated_arg=mutated_arg
            )

    def copy_node_for_reinplaced_arg(node, mutated_arg):
        # [FUNCTION DOC] HELPER: Finds the epilogue copy_ node that writes this op's result back to mutated_arg, directly or via views (the via-views
        # case is verified layout preserving, else returns None). Returns None if there is none.
        copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
        if copy_node is None:
            copy_node = copy_args_to_copy_nodes_via_views.get((mutated_arg, node))
            if copy_node is not None:
                copy_dst = copy_node.args[0]
                copy_src = copy_node.args[1]
                if not (
                    isinstance(copy_dst, torch.fx.Node)
                    and isinstance(copy_src, torch.fx.Node)
                    and _is_layout_preserving_view_copy_back(
                        copy_dst, copy_src, mutated_arg, node
                    )
                ):
                    return None
        return copy_node

    def all_can_inplace(node, mutated_args):
        # [FUNCTION DOC] HELPER: True if can_inplace holds for every arg in mutated_args.
        return all(can_inplace(node, arg) for arg in mutated_args)

    def log_inplace_results(
        node_name,
        old_tensors_to_clone,
        tensors_to_clone,
        missed_args,
        missed_nodes,
        trigger,
    ):
        # [FUNCTION DOC] HELPER: Logs (info) which tensors were attempted, kept as clones, and missed, with the total static-shape bytes of missed
        # opportunities, and records missed count and bytes in ReinplaceCounters for the trigger. Side effects: logging and counter updates.
        # Total size of possibly_missed_reinplacing_opportunities for tensors with static shapes.
        missed_bytes = 0

        def bytes(node):
            # [FUNCTION DOC] HELPER: Size in bytes of the node's fake tensor value (element size times numel) if statically known, else 0.
            t = node.meta.get("val", None)
            if (
                t is not None
                and isinstance(t.element_size(), int)
                and isinstance(t.numel(), int)
            ):
                return t.element_size() * t.numel()
            else:
                return 0

        for node in missed_nodes:
            if isinstance(node, (list, tuple)):
                for n in node:
                    missed_bytes += bytes(n)
            else:
                missed_bytes += bytes(node)

        log.info(
            "For node %s, attempted to reinplace %s. We were unable to reinplace %s; "
            "%s (if non-empty) are possible missed reinplacing opportunities that may be bad for "
            "memory usage and performance. Total size of missed opportunities with static shapes is"
            " : %s bytes.",
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_bytes,
        )

        ReinplaceCounters.add_missed_opportunities(trigger, len(missed_args))
        ReinplaceCounters.add_missed_bytes(trigger, missed_bytes)

    replace_dict: dict[torch.fx.Node, torch.fx.Node] = {}

    def reinplace_and_refine_tensors_to_clone(
        old_tensors_to_clone, kwargs, node_name, trigger
    ):
        # [FUNCTION DOC] Given argument names/indices of an auto_functionalized or triton node, decides which can be reinplaced and returns the list that
        # must still be cloned. Does not reinplace two args sharing storage with one already reinplaced. For each reinplaceable arg, records epilogue copy_ nodes
        # (and, except for auto_functionalize v2, getitem users selecting the arg) in replace_dict, and logs results.
        # Raises AssertionError if an arg is missing from kwargs.
        tensors_to_clone: list[str] = []
        storage_of_reinplaced_args = OrderedSet[int | None]()

        # Those used to count possibly_missed_reinplacing_opportunities
        missed_nodes = []
        missed_args = []

        # TODO this logic can be made more precise using _overlap
        def tensor_with_same_storage_already_reinplaced(arg):
            # [FUNCTION DOC] HELPER: True if the storage of the arg (each element if a list/tuple) is already in storage_of_reinplaced_args.
            # KNOWN ISSUE: the non-list branch reads the enclosing loop variable mutated_arg rather than the arg parameter; behaves the same only because callers pass that value.
            if isinstance(arg, (list, tuple)):
                return any(
                    get_node_storage(a) in storage_of_reinplaced_args for a in arg
                )
            return get_node_storage(mutated_arg) in storage_of_reinplaced_args

        for arg in old_tensors_to_clone:
            if arg not in kwargs:
                raise AssertionError(f"expected {arg} to be in kwargs")

            mutated_arg = kwargs[arg]

            # Let's say we have:
            # - op(x, y) that mutates both x and y
            # - new_x, new_y = functional_op(x, y) is the functional variant
            # If we are presented with functional_op(x, x), we must not reinplace
            # this into op(x, x), because then it would be writing to the same Tensor.
            # Instead, it's OK to reinplace one of them and to clone the other:
            # >>> y = x.clone()
            # >>> op(x, y)
            # This also applies if we have views: functional_op(x, x[0])
            # should not reinplace into op(x, x[0]).
            should_attempt_reinplace = not tensor_with_same_storage_already_reinplaced(
                mutated_arg
            )
            if should_attempt_reinplace and can_inplace(node, mutated_arg):
                # In general, we probably do not need those optimizations.
                copy_node = copy_node_for_reinplaced_arg(node, mutated_arg)
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                if trigger != ReInplaceTrigger.AUTO_FUNC_V2:
                    for user in node.users:
                        # For auto_functionalize_v2, arg is the index of the base, where base at index i corresponds to
                        # output at index size(out)+i.
                        # This used to compare string with integers before for auto_functionalize_v2. Not sure
                        # if it was needed for inplaceable_triton_ops?
                        if user.target is operator.getitem and user.args[1] == arg:
                            replace_dict[user] = mutated_arg

                if isinstance(mutated_arg, (list, tuple)):
                    for a in mutated_arg:
                        storage_of_reinplaced_args.add(get_node_storage(a))
                else:
                    storage_of_reinplaced_args.add(get_node_storage(mutated_arg))
            else:
                if should_attempt_reinplace:
                    missed_args.append(arg)
                    missed_nodes.append(mutated_arg)

                tensors_to_clone.append(arg)

        log_inplace_results(
            node_name,
            old_tensors_to_clone,
            tensors_to_clone,
            missed_args,
            missed_nodes,
            trigger,
        )
        return tensors_to_clone

    for node in graph.nodes:
        if (inplaceable_op := inplaceable_ops.get(node.target)) is not None:
            # Check if ALL mutated args can be inplaced
            # Only convert if we don't need to clone any tensor
            mutated_args = [node.args[idx] for idx in inplaceable_op.mutated_args]
            if all_can_inplace(node, mutated_args) and inplaceable_op.extra_check(node):
                for mutated_arg in mutated_args:
                    copy_node = copy_node_for_reinplaced_arg(node, mutated_arg)
                    if copy_node is not None:
                        replace_dict[copy_node] = copy_node.args[0]
                node.target = inplaceable_op.inplace_op
        elif node.target is torch.ops.higher_order.auto_functionalized_v2:
            _mutable_op = node.args[0]
            kwargs = node.kwargs

            all_bases = kwargs["_all_bases"]
            bases_to_clone = range(len(all_bases))
            base_tensors_dct = dict(enumerate(all_bases))
            new_bases_to_clone: list[int] = reinplace_and_refine_tensors_to_clone(
                bases_to_clone,
                base_tensors_dct,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V2,
            )
            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = new_bases_to_clone
        elif node.target is torch.ops.higher_order.auto_functionalized:
            _mutable_op = node.args[0]
            from torch._higher_order_ops.auto_functionalize import get_mutable_args

            tensors_to_clone, _ = get_mutable_args(_mutable_op)
            # Don't try to reinplace Tensor | None args that are None.
            tensors_to_clone = [
                t for t in tensors_to_clone if node.kwargs[t] is not None
            ]
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                tensors_to_clone,
                node.kwargs,
                _mutable_op._name,
                ReInplaceTrigger.AUTO_FUNC_V1,
            )

            # Stash the metadata. There is a pass later on where we decompose
            # auto_functionalized into clones + a mutable op; this metadata
            # tells the decomp to only clone the following inputs
            node.meta["only_clone_these_tensors"] = tensors_to_clone
        elif node.target is torch.ops.higher_order.with_effects:
            # Handle effectful ops wrapped with with_effects
            # args[0] is the token, args[1] is the inner op, args[2:] are the op's args
            inner_op = node.args[1]
            log.debug(
                "reinplace: checking with_effects node with inner_op=%s", inner_op
            )
            if (inplaceable_op := inplaceable_ops.get(inner_op)) is not None:
                log.debug("reinplace: found inplaceable_op for %s", inner_op)
                # Get the mutated arg indices, offset by 2 (token + op)
                mutated_arg_indices = inplaceable_op.mutated_args

                # Build flat list of tensors for can_inplace check
                # and a mapping of output index -> replacement tensor(s)
                mutated_tensors_flat = []
                output_idx_to_replacement: dict[int, Any] = {}

                for position, idx in enumerate(mutated_arg_indices):
                    actual_idx = idx + 2  # offset for token and op
                    if actual_idx >= len(node.args):
                        raise AssertionError(
                            f"mutated arg idx {actual_idx} out of range {len(node.args)}"
                        )
                    arg = node.args[actual_idx]

                    # Output index is position + 1 (index 0 is the token)
                    output_idx = position + 1
                    output_idx_to_replacement[output_idx] = arg

                    # Flatten for can_inplace check
                    if isinstance(arg, (list, tuple)):
                        mutated_tensors_flat.extend(arg)
                    else:
                        mutated_tensors_flat.append(arg)

                # Check if all mutated args can be inplaced
                can_inplace_all = all_can_inplace(node, mutated_tensors_flat)

                log.debug(
                    "reinplace with_effects: mutated_tensors=%s, can_inplace_all=%s",
                    [str(a) for a in mutated_tensors_flat],
                    can_inplace_all,
                )

                if can_inplace_all and inplaceable_op.extra_check(node):
                    log.debug(
                        "reinplace with_effects: converting %s -> %s",
                        inner_op,
                        inplaceable_op.inplace_op,
                    )
                    # Update the inner op to inplace version
                    node.update_arg(1, inplaceable_op.inplace_op)

                    # The output structure changes: functional returns (token, tensors),
                    # inplace returns (token, None). We need to redirect tensor uses
                    # to the input tensors.

                    def get_index_from_node(n):
                        # [FUNCTION DOC] HELPER: Returns the index argument of a getitem node, or None if n is not a getitem.
                        """Extract the index from a getitem node."""
                        if n.target is operator.getitem:
                            return n.args[1]
                        return None

                    def is_getitem_node(n, parent):
                        # [FUNCTION DOC] HELPER: True if n is a getitem node indexing into parent.
                        """Check if node n is a getitem indexing into parent."""
                        return n.target is operator.getitem and n.args[0] is parent

                    def replace_and_collect(current_node, replacement_tensors):
                        # [FUNCTION DOC] HELPER: Recursively fills replace_dict so getitem results of a with_effects node are replaced by the input tensors they correspond to.
                        # Children are recorded before parents. Leaf nodes map to the single replacement; otherwise getitem users are matched by integer index into the replacement
                        # list. Returns True if all children were replaced.
                        """
                        Collect replacements for getitem nodes into replace_dict.
                        Nodes are added in child-first order so children are erased before parents.
                        """
                        # Find all users that are getitem nodes indexing into current_node
                        getitem_users = [
                            u
                            for u in current_node.users
                            if is_getitem_node(u, current_node)
                        ]

                        if not getitem_users:
                            # Leaf node - add to replace_dict with actual replacement
                            if isinstance(replacement_tensors, (list, tuple)):
                                if len(replacement_tensors) == 1:
                                    replace_dict[current_node] = replacement_tensors[0]
                                    return True
                                else:
                                    # Multiple tensors but no indexing - can't replace
                                    return False
                            else:
                                replace_dict[current_node] = replacement_tensors
                                return True

                        # Process children first (so they're added to replace_dict before parent)
                        all_children_replaced = True
                        first_replacement = None
                        for getitem_user in getitem_users:
                            idx = get_index_from_node(getitem_user)
                            if idx is None or not isinstance(idx, int):
                                all_children_replaced = False
                                continue

                            if not isinstance(replacement_tensors, (list, tuple)):
                                all_children_replaced = False
                                continue

                            if idx >= len(replacement_tensors):
                                all_children_replaced = False
                                continue

                            if first_replacement is None:
                                first_replacement = replacement_tensors[idx]

                            if not replace_and_collect(
                                getitem_user, replacement_tensors[idx]
                            ):
                                all_children_replaced = False

                        # Add this node to replace_dict after children (even if it has non-getitem users)
                        # Non-getitem users will have their input replaced via replace_all_uses_with
                        if all_children_replaced and first_replacement is not None:
                            replace_dict[current_node] = first_replacement

                        return all_children_replaced

                    # Find getitem nodes that extract tensor results
                    # Use the output_idx_to_replacement mapping built above
                    for user in list(node.users):
                        if not is_getitem_node(user, node):
                            continue
                        idx = get_index_from_node(user)
                        if idx is None or idx not in output_idx_to_replacement:
                            continue
                        replacement = output_idx_to_replacement[idx]
                        replace_and_collect(user, replacement)
        elif node.target in inplaceable_triton_ops:
            kernel_idx = node.kwargs["kernel_idx"]
            kernel = kernel_side_table.get_kernel(kernel_idx)
            from triton.runtime.autotuner import Autotuner
            from triton.runtime.jit import JITFunction

            if isinstance(kernel, JITFunction):
                kernel_name = kernel.fn.__name__
            elif isinstance(kernel, Autotuner):
                if config.is_fbcode():
                    # Autotuner has different implementations for AMD and NV
                    if torch.version.hip is None:
                        kernel_name = kernel.base_fn.__name__
                    else:
                        kernel_name = kernel.fn.__name__
                else:
                    kernel_name = kernel.base_fn.__name__
            else:
                raise AssertionError("Unknown triton kernel type")

            # inplaceable_triton_ops take an additional argument called
            # tensors_to_clone which contain a list of tensors to clone
            # This pass iterates over them and sees which ones are safe
            # to eliminate (i.e. no longer need the clones)
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                node.kwargs["tensors_to_clone"],
                node.kwargs["kwargs"],
                kernel_name,
                ReInplaceTrigger.TRITON_OPS,
            )

            kwargs = dict(node.kwargs)
            kwargs["tensors_to_clone"] = tensors_to_clone
            node.kwargs = immutable_dict(kwargs)
            if "eager_input_vals" in node.meta:
                # We changed the kwargs, so we need to update eager_input_vals
                # to something sane.
                args, kwargs = node.meta["eager_input_vals"]
                new_kwargs = {**kwargs}
                new_kwargs["tensors_to_clone"] = immutable_list(tensors_to_clone)
                new_kwargs = immutable_dict(new_kwargs)
                node.meta["eager_input_vals"] = (args, new_kwargs)
        elif (inplaceable_op := inplaceable_foreach_ops.get(node.target)) is not None:
            mutated_args = node.args[inplaceable_op.mutated_arg]

            if not all((arg, node) in copy_args_to_copy_nodes for arg in mutated_args):
                continue

            if can_inplace(node, mutated_args):
                for arg in mutated_args:
                    copy_node = copy_args_to_copy_nodes[(arg, node)]
                    replace_dict[copy_node] = copy_node.args[0]

                node.target = inplaceable_op.inplace_op
    for node, replacement in replace_dict.items():
        while replacement in replace_dict:
            replacement = replace_dict[replacement]
        replace_dict[node] = replacement

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)


def reinplace_inplaceable_ops(
    fake_tensor_updater: torch._inductor.fx_utils.FakeTensorUpdater,
    graph: torch.fx.Graph,
) -> None:
    # [FUNCTION DOC] Public entry: under the python dispatcher, canonicalizes view scatter ops, refreshes fake tensor alias info with fake_tensor_updater,
    # runs the core reinplace pass, then decomposes generalized scatters. Mutates graph.
    with enable_python_dispatcher():
        canonicalize_view_scatter_ops(graph)
        # canonicalize_view_scatter_ops adds new operations to the graph.
        # We run fake_tensor_updater to update the alias information.
        # Correct alias information is required for `reinplace_inplaceable_ops_core`.
        fake_tensor_updater.incremental_update()
        reinplace_inplaceable_ops_core(graph)
        decompose_generalized_scatter(graph)
