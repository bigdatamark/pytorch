# [MODULE DOC] Post-functionalization FX graph pass that rewrites complex-valued
# ops into real-valued ops. Main entry point is decompose_complex_in_graph; the
# private helpers detect complex tensors, wrap/unwrap them as ComplexTensor
# subclass instances, and reject in-place mutation of complex tensors. Retracing
# uses make_fx under WrapComplexMode. No import-time side effects.
"""
Graph pass to decompose complex-valued operations into real-valued operations.

This pass runs AFTER functionalization, so the graph is purely functional (no
mutations, no aliasing). It retraces the graph with ComplexTensor dispatch active,
which decomposes each complex op into real-valued sub-ops. The graph signature
(input/output count and dtypes) is preserved: complex inputs get unpacked via
aten.real/aten.imag at the top, and complex outputs get repacked via aten.complex
at the bottom.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.complex_tensor import ComplexTensor, WrapComplexMode
from torch.fx.experimental.proxy_tensor import make_fx


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._ops import OpOverload


def _can_wrap(value: Any) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if value is a plain (non-ComplexTensor)
    # tensor with a complex dtype, i.e. one that should be wrapped. Anything else
    # (non-tensors, real tensors, already-wrapped ComplexTensors) gives False.
    return (
        isinstance(value, Tensor)
        and value.dtype.is_complex
        and not isinstance(value, ComplexTensor)
    )


def _maybe_wrap(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: If value is a wrappable complex tensor, returns a
    # ComplexTensor built from its real and imaginary parts; otherwise returns
    # value unchanged.
    if _can_wrap(value):
        return ComplexTensor(torch.real(value), torch.imag(value))
    return value


def _maybe_unwrap(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: If value is a ComplexTensor, converts it back to an
    # ordinary interleaved complex tensor; otherwise returns value unchanged.
    if isinstance(value, ComplexTensor):
        return value.as_interleaved()
    return value


def _has_complex(gm: fx.GraphModule, flat_args: list[Any]) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if any flat input is a wrappable complex
    # tensor, or any call_function node in the graph has a complex tensor in its
    # recorded "val" metadata. Pure query, no mutation.
    for arg in flat_args:
        if _can_wrap(arg):
            return True
    for node in gm.graph.nodes:
        if node.op == "call_function":
            val = node.meta.get("val")
            if _can_wrap(val):
                return True
    return False


def _aliased_input_indices(op: torch._ops.OpOverload) -> list[int]:
    # [FUNCTION DOC] HELPER: Using the op schema, collects the alias sets that
    # returns belong to, then returns the positions of arguments whose alias set
    # overlaps them (arguments the output may alias). Empty list if no return
    # carries alias info.
    schema = op._schema
    ret_aliases = set()
    idxs = []

    for ret in schema.returns:
        ret_alias = ret.alias_info
        if ret_alias is None:
            continue
        ret_aliases.update(ret_alias.before_set)

    if len(ret_aliases) == 0:
        return idxs

    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None:
            continue
        if not arg.alias_info.before_set.isdisjoint(ret_aliases):
            idxs.append(i)

    return idxs


def _collect_storage_aliases(node: fx.Node) -> set[fx.Node]:
    # [FUNCTION DOC] HELPER: Walks backward from node through OpOverload nodes,
    # following the arguments each op may alias (per schema), and returns the set
    # of all visited nodes including node itself. Non-OpOverload nodes are
    # included but not expanded. Currently unused within this module.
    seen: set[fx.Node] = set()
    stack: list[fx.Node] = [node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if not isinstance(n.target, torch._ops.OpOverload):
            continue
        idxs = _aliased_input_indices(n.target)
        for idx in idxs:
            src = n.args[idx]
            if isinstance(src, fx.Node):
                stack.append(src)
    return seen


def _assert_no_incorrect_aliasing_mutation(gm: fx.GraphModule) -> None:
    # [FUNCTION DOC] HELPER: Scans graph nodes and raises RuntimeError if any
    # aten copy_ call has a complex-dtype destination (first argument, read from
    # its "val" metadata), since in-place mutation of complex tensors is
    # unsupported in a compiled region. Otherwise returns None.
    # KNOWN ISSUE: the docstring describes three checks (incorrect aliases,
    # mutated complex inputs, input/output aliasing) and mentions an
    # INCORRECT_ALIASING_OPS list, but only the copy_ on complex check exists.
    """
    In this function, we check that the compiled graph respects aliasing semantics.

    There are three cases in which this check fires:
        1. A complex tensor is mutated in place via an incorrect alias.
           The ops that produce incorrect aliases are listed in `INCORRECT_ALIASING_OPS`.
        2. A complex input is mutated in place. This is not allowed because complex inputs
           decompose to real and imaginary parts, which are made contiguous for performance.
           Therefore, any complex input passed in does not alias the original tensor.
        3. A complex input aliases an output. This is not allowed as the input and output
           tensors can't alias each other because of 2.

    In all three cases, we raise an error if possible incorrectness is detected.
    """
    for n in gm.graph.nodes:
        if (
            isinstance(n.target, torch._ops.OpOverload)
            and n.target.overloadpacket == torch.ops.aten.copy_
            and n.args[0].meta["val"].dtype.is_complex
        ):
            raise RuntimeError(
                "Mutating a complex tensor in place is not allowed in a `torch.compile` region."
            )


def decompose_complex_in_graph(
    gm: fx.GraphModule,
    flat_args: list[Any],
    decompositions: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> fx.GraphModule:
    # [FUNCTION DOC] Entry point of the pass. If neither the flat inputs nor graph
    # node metadata contain complex tensors, returns gm untouched. Otherwise
    # validates there is no complex in-place mutation (may raise RuntimeError),
    # then retraces gm with make_fx under WrapComplexMode: complex inputs are
    # wrapped as ComplexTensor (real/imag parts), the graph is interpreted, and
    # outputs are unwrapped back to complex. Optional decompositions is a table
    # passed to make_fx. Returns a new GraphModule with the same signature.
    if not _has_complex(gm, flat_args):
        return gm

    _assert_no_incorrect_aliasing_mutation(gm)

    def wrapper(*args: Any) -> Any:
        # [FUNCTION DOC] HELPER: Traced function. Wraps complex args, runs gm with
        # an fx Interpreter inside WrapComplexMode so complex ops decompose, and
        # returns the result pytree with ComplexTensors converted back to complex.
        wrapped = tuple(_maybe_wrap(a) for a in args)
        with WrapComplexMode():
            result = fx.Interpreter(gm).run(*wrapped)
        return pytree.tree_map(_maybe_unwrap, result)

    with fx_traceback.preserve_node_meta():
        return make_fx(wrapper, decomposition_table=decompositions)(*flat_args)
