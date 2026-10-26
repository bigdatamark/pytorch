# [MODULE DOC] FX graph helpers used by functorch/AOT compilation: common subexpression
# elimination (fx_graph_cse) that is aware of randomness, aliasing of outputs, mutation
# regions and unbacked symbols; plus small utilities to hoist getitem nodes, strip op overloads,
# and fetch placeholders/outputs. Import-time side effects: defines the aten alias and the
# rand_ops list of random ATen ops that CSE must not merge.
from __future__ import annotations

import dataclasses
import operator
import struct
from typing import Any, TYPE_CHECKING

import sympy

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from torch._ops import OpOverloadPacket
    from torch.utils._pytree import TreeSpec

aten = torch.ops.aten


@dataclasses.dataclass(frozen=True)
class _ScalarKey:
    # [CLASS DOC] HELPER: Frozen hashable key for a float or complex scalar in CSE tokens.
    # Holds a type tag and the big-endian packed bytes so equality/hash follow bit patterns
    # (nan equals itself, -0.0 differs from 0.0).
    tag: str
    bits: bytes


def _normalize_cse_arg(val: Any) -> Any:
    # [FUNCTION DOC] HELPER: Maps an exact float or complex value to a _ScalarKey built from
    # its packed bits; any other value (including subclasses and ints) is returned unchanged.
    # Python float hash/eq is not value-identity: nan != nan (hash(nan) is
    # id-based) while -0.0 == 0.0 and hashes equal. Key by bit pattern instead.
    if type(val) is float:
        return _ScalarKey("float", struct.pack(">d", val))
    if type(val) is complex:
        return _ScalarKey("complex", struct.pack(">dd", val.real, val.imag))
    return val


def get_aten_target(node: fx.Node) -> OpOverloadPacket | Callable[..., Any] | str:
    # [FUNCTION DOC] Returns the overload packet of the node's target if it has one, otherwise
    # the target itself.
    if hasattr(node.target, "overloadpacket"):
        return node.target.overloadpacket
    return node.target


rand_ops = [
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
]


# return a new copy of torch.fx.graph.Graph with CSE applied to the input graph
def fx_graph_cse(
    fx_g: torch.fx.graph.Graph,
    extra_node_key: Callable[[fx.Node], Hashable] | None = None,
) -> fx.Graph:
    # [FUNCTION DOC] Returns a new FX graph equal to fx_g with common subexpressions removed.
    # Placeholder/output/get_attr, impure, random, aten.empty, output-aliasing, mutation-base,
    # and unbacked-symbol nodes are copied as is. Other nodes get a token (target, optional
    # extra_node_key result, normalized args/kwargs, stream/mempool context); a node is
    # replaced by an earlier one with an equal token in the same mutation region, otherwise
    # it replaces the earlier entry if the region differs. Calls compute_mutation_region_ids
    # on fx_g (mutates node metadata). Raises AssertionError if the last node is not output.
    new_graph = fx.Graph()
    env: dict[
        fx.Node, fx.Node
    ] = {}  # map from node in the old graph to node in the new graph
    hash_env: dict[
        tuple[Any, Hashable | None, int], fx.Node
    ] = {}  # map from hash to a node in the new graph
    token_map: dict[
        tuple[Any, Hashable | None, int], dict[str, Any]
    ] = {}  # map from hash to token

    from torch._inductor.pattern_matcher import (
        compute_mutation_region_ids,
        same_mutation_regions,
    )

    compute_mutation_region_ids(fx_g)  # type: ignore[arg-type]

    # Make a set of separate storages returned from the output, which will be preserved
    # when pruning.  This prevents us from deduplicating returned tensors which have
    # experienced identical operations, but are separate data structures in eager mode.
    output_node: fx.Node = list(fx_g.nodes)[-1]
    if output_node.op != "output":
        raise AssertionError(
            f"expected output_node.op to be 'output', got '{output_node.op}'"
        )

    def checkable_node(node: fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: True if the node's meta val is a tensor whose untyped storage
        # can be obtained (False for non-tensors or storage-less tensors).
        """We can evaluate only nodes that represent tensors with defined storage."""
        if "val" not in node.meta or not isinstance(node.meta["val"], torch.Tensor):
            return False

        try:
            node.meta["val"].untyped_storage()
        except NotImplementedError:
            return False

        return True

    def custom_context_key(node: fx.Node) -> tuple[int, int | None, int | None]:
        # [FUNCTION DOC] HELPER: Returns (stream, mempool, mempool_device) from the node's
        # custom meta (stream defaults to 0, others to None) so nodes in different contexts
        # are not merged.
        custom = node.meta.get("custom", {})
        return (
            custom.get("stream", 0),
            custom.get("mempool"),
            custom.get("mempool_device"),
        )

    output_storages = {
        StorageWeakRef(n.meta["val"].untyped_storage())
        for n in output_node.all_input_nodes
        if checkable_node(n)
    }
    nodes_that_alias_outputs = {
        n
        for n in fx_g.nodes
        if checkable_node(n)
        and StorageWeakRef(n.meta["val"].untyped_storage()) in output_storages
    }

    # A node used as the mutable base of a functional-wrapper op is a distinct
    # mutable buffer. Two identical factory ops feeding distinct bases (e.g. the
    # forward and backward ``aten.full`` from a pair of ``ones_like`` calls) must
    # not be CSE-merged, or the later mutation aliases a buffer that still has a
    # live downstream view and reinplacing fails (mirrors the aten.empty
    # exclusion below). See pytorch/pytorch#170160.
    from torch._higher_order_ops.auto_functionalize import get_mutable_args
    from torch._higher_order_ops.triton_kernel_wrap import (
        triton_kernel_wrapper_functional,
    )

    def _add_base(node: Any, bases: set[fx.Node]) -> None:
        # [FUNCTION DOC] HELPER: Adds node (or each element if it is a list/tuple) to the bases
        # set, ignoring anything that is not an fx Node.
        for b in node if isinstance(node, (list, tuple)) else (node,):
            if isinstance(b, fx.Node):
                bases.add(b)

    nodes_used_as_mutation_base: set[fx.Node] = set()
    for n in fx_g.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized_v2
    ):
        # v2 lists its mutable buffers explicitly in ``_all_bases``.
        for base in n.kwargs.get("_all_bases", ()):
            _add_base(base, nodes_used_as_mutation_base)
    for n in fx_g.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized
    ):
        # v1 has no ``_all_bases``; the mutated tensors are the args named by the
        # wrapped op's schema (``_mutable_op`` is the first positional).
        mutable_op = n.args[0]
        mutable_args_names, _ = get_mutable_args(mutable_op)
        for name in mutable_args_names:
            _add_base(n.kwargs.get(name), nodes_used_as_mutation_base)
    for n in fx_g.find_nodes(
        op="call_function", target=triton_kernel_wrapper_functional
    ):
        inner_kwargs = n.kwargs.get("kwargs", {})
        for name in n.kwargs.get("tensors_to_clone", ()):
            _add_base(inner_kwargs.get(name), nodes_used_as_mutation_base)

    for n in fx_g.nodes:
        # The placeholder, output, and get_attr nodes are copied to the new graph without change
        # do not CSE away random operations
        if (
            n.op == "placeholder"
            or n.op == "output"
            or n.op == "get_attr"
            or n.is_impure()
            or get_aten_target(n) in rand_ops
            # aten.empty is non-deterministic, so don't CSE it.
            # Also, aten.empty is almost always fusible into its consumer,
            # so it's not worth CSEing.
            or get_aten_target(n) is aten.empty
            or n in nodes_that_alias_outputs
            # Keep distinct functional-wrapper mutation bases independent.
            or n in nodes_used_as_mutation_base
            # This CSE pass currently doesn't handle re-propagation of unbacked
            # meta where it'll sometimes eliminate a _local_scalar_dense but not
            # replace the meta of downstream users. eg. one bug we've seen is:
            #
            # _local_scalar_dense_11: "Sym(u14)" = torch.ops.aten._local_scalar_dense.default(select_10);
            # sym_sum_2: "Sym(u19 + u20 + u21)" = torch.sym_sum((_local_scalar_dense_11, _local_scalar_dense_12, _local_scalar_dense_13))
            #
            # Notice how _local_scalar_dense_11 is u14 but sym_sum_2's meta is incorrectly the old
            # pre-cse value of u19.
            or (
                "val" in n.meta
                and isinstance(n.meta["val"], sympy.Symbol)
                and free_unbacked_symbols(n.meta["val"])
            )
        ):
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
            # substitute args and kwargs members to their mapping in env if exists
            # specs can be used to reconstruct nested list/dictionaries
            def substitute(
                arg_list: list[Any] | tuple[Any, ...],
            ) -> tuple[tuple[Any, ...], TreeSpec]:
                # [FUNCTION DOC] HELPER: Flattens a nested args/kwargs structure, replaces old
                # graph nodes by their new-graph nodes, symbolic ints/bools/floats by their
                # underlying node, and floats/complex by bit-pattern keys. Returns the flat
                # tuple and the tree spec for reconstruction.
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                    arg_list[i] = _normalize_cse_arg(arg_list[i])
                return tuple(arg_list), spec

            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)

            # each token corresponds to a unique node
            # nodes with the same token can be substituted
            extra_key = extra_node_key(n) if extra_node_key is not None else None
            token = {
                "target": n.target,
                "extra_key": extra_key,
                "args": args,
                "args_spec": args_spec,
                "kwargs": kwargs,
                "kwargs_spec": kwargs_spec,
                "custom_context": custom_context_key(n),
            }

            # hash substituted args to a number, do not hash specs because specs are not hashable
            # We need to add type into hash to avoid situations like:
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash(
                (tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs))
            )
            hash_val = (n.target, extra_key, hash_arg)

            # check if a node has a substitute and can be eliminated
            hash_val_in_hash_env = hash_val in hash_env
            overwrite_due_to_mutation = False
            if hash_val_in_hash_env and token_map[hash_val] == token:
                duplicate_n_prev = hash_env[hash_val]
                if same_mutation_regions(n, duplicate_n_prev):
                    env[n] = duplicate_n_prev
                    continue
                else:
                    # any futures duplicates should replace with n, not duplicate_n_prev
                    overwrite_due_to_mutation = True

            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if overwrite_due_to_mutation or not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token

    return new_graph


def raise_getitems(gm: fx.GraphModule) -> fx.GraphModule:
    # [FUNCTION DOC] Moves every operator.getitem node to directly after its single input node
    # (processing in reverse to keep relative order), recompiles gm, and returns it. Mutates gm
    # in place. Raises AssertionError if a getitem node has other than one input node.
    # Pre-create a list of nodes to iterate over, as modifying the node order
    # during the loop can lead to infinite loops if not handled properly.
    getitem_nodes = list(
        gm.graph.find_nodes(op="call_function", target=operator.getitem)
    )

    # loop through getitem nodes in the graph and raise them to the parent node
    # in reverse order to preserve their original relative order
    for node in reversed(getitem_nodes):
        if len(node.all_input_nodes) != 1:
            raise AssertionError(
                f"expected node {node.name} to have 1 input node, got {len(node.all_input_nodes)}"
            )
        parent = node.all_input_nodes[0]
        parent.append(node)

    gm.recompile()
    return gm


def strip_overloads(gm: fx.GraphModule) -> None:
    # [FUNCTION DOC] Replaces each OpOverload node target with its overload packet in place and
    # recompiles gm. Returns nothing.
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def get_placeholders(graph: fx.Graph) -> list[Any]:
    # [FUNCTION DOC] Returns the placeholder nodes of the graph.
    return graph.find_nodes(op="placeholder")


def get_outputs(graph: fx.Graph) -> list[fx.Node]:
    # [FUNCTION DOC] Returns the flattened leaves of the first output node's args. Raises
    # AssertionError if the graph has no output node.
    for node in graph.find_nodes(op="output"):
        return pytree.tree_leaves(node.args[0])
    raise AssertionError("No output node found")
