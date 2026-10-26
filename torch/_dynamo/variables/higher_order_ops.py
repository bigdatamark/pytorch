# [MODULE DOC] Dynamo support for tracing higher-order operators (HOPs) such as cond, while_loop,
# map, scan, invoke_subgraph, activation checkpoint, autograd.Function and flex attention.
# Main contents: speculate_subgraph and its auto-output-flattening variant (trace a callable into a
# separate FX subgraph), graph input merging/reordering helpers, aliasing/mutation checks, output
# spec handling, and one VariableTracker subclass per HOP registered in a name-to-class mapping.
# Import-time side effects: creates module loggers, type variables, and registers HOP variable
# classes (including applying the add_hop_context class decorator to subclasses).
"""
This module contains classes and utilities for handling higher-order operators in Dynamo.
It provides functionality for tracing and transforming control flow constructs like
conditions (torch.cond), loops (torch.while_loop), maps (torch.ops.higher_order.map),
and other higher-order operations.

The module includes specialized VariableTracker classes for different types of
higher-order operations, along with utilities for:
- Speculating and capturing subgraphs
- Managing control flow
- Handling autograd function applications
- Supporting function transformations
- Processing activation checkpoints

These classes work together to enable Dynamo to correctly trace and compile code
containing complex control flow patterns and higher-order functions while preserving
their semantic behavior.
"""

import contextlib
import copy
import functools
import inspect
import itertools
import logging
import traceback
import types
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast, get_args, Literal, Optional, TYPE_CHECKING, Union

import torch._C
import torch.fx
import torch.nn
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import get_fake_value
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.ctx_manager import RepararametrizeModuleContextVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.script_object import CustomClassObjectVariable
from torch._dynamo.variables.tensor import SymNodeVariable, TensorVariable
from torch._guards import Source
from torch._higher_order_ops.flex_gemm import FLEX_GEMM_OP_ALIASES
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
from torch._ops import HigherOrderOperator
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.proxy import Proxy
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet

from .. import graph_break_hints, variables
from ..exc import (
    FakeTensorObservedException,
    ObservedException,
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
)
from ..source import AttrSource, DictGetItemSource
from ..utils import proxy_args_kwargs, set_example_value, unpack_iterable
from .base import VariableTracker
from .dicts import ConstDictVariable
from .lazy import LazyVariableTracker
from .lists import ListVariable, TupleVariable
from .sets import SetVariable


if TYPE_CHECKING:
    from torch._dynamo.output_graph import SubgraphTracer
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from . import AutogradFunctionContextVariable

from collections.abc import Generator, Iterable
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")
HOP_VT_Alias = TypeVar("HOP_VT_Alias", bound="TorchHigherOrderOperatorVariable")

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

# How speculate_subgraph constructs subgraph placeholders from sub_args. See
# NOTE [argument `set_subgraph_inputs`] below for the meaning of each value. This
# is the single source of truth for the runtime asserts that validate the value.
SetSubgraphInputs = Literal[
    "automatic", "automatic_with_forced_inputs", "flatten_manual", "manual"
]


@dataclass
class OutputSpec:
    # [CLASS DOC] Dataclass describing a speculated subgraph output: the pytree treespec, an
    # optional per-leaf boolean mask marking constant leaves removed from the graph output, the
    # removed constant values (same length as the mask), and a count of extra intermediate outputs.
    """
    Contains the treespec of the output of the speculated subgraph, and the
    information to mask out the constant values from the output during
    flattening and inserting them back during unflattening. Cleaning up
    constants from the graph makes the graph simpler for AOTDispatcher and
    Inductor.
    """

    treespec: pytree.TreeSpec
    # list of True/False to identify the locations of const values in the
    # subgraph output. True means that value at that index is a constant.
    masks_to_filter_const_values: list[bool] | None = None
    # The actual constant values that were present in the subgraph output. Note
    # that this is the same length as the mask, we just look at the indices
    # where mask is True.
    const_values: list[Any] | None = None
    # Number of intermediate nodes that are also made subgraph outputs.
    num_intermediate_nodes_as_outputs: int = 0

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Dataclass validation hook. Raises AssertionError if both the
        # const mask and the const values are present but differ in length.
        if (
            self.masks_to_filter_const_values is not None
            and self.const_values is not None
        ):
            if len(self.masks_to_filter_const_values) != len(self.const_values):
                raise AssertionError(
                    "masks_to_filter_const_values and const_values must have the same length"
                )


@dataclass(frozen=True)
class SubgraphTracingInfo:
    # [CLASS DOC] Frozen dataclass of metadata observed while tracing a HOP subgraph: the user
    # stack of the first externally visible side effect (or none) and the ordered set of sources
    # accessed through the variable builder during the trace.
    """Properties observed during subgraph tracing.

    Expandable container for metadata collected while tracing a HOP subgraph.
    Future additions may include aliasing info, mutation info, etc.
    """

    # User code stack at the point where the first externally-visible side
    # effect was detected.  None means no side effect was observed.
    side_effect_stack: traceback.StackSummary | None = None
    # All sources accessed via VariableBuilder during the subgraph trace.
    traced_sources: OrderedSet[Any] = field(default_factory=OrderedSet)


# This function is a syntax sugar for creating a dummy new subtracer so that
# newly added nodes are added to a separate subgraph in this subtracer instead of affecting
# the main graph. This is useful for creating sample inputs for tracing the subgraph.
# For example, in FlexAttentionHigherOrderVariable, we want to create several scalars
# to trace the score_mod function but we don't want the operators that creates the scalar to
# show up in the graph, we could this function to discard the graph changes.
# Example usage:
# with discard_graph_changes():
#   sample_input= create_sample_inputs()
# speculate_subgraph(tx, f, sample_inputs, {})
@contextlib.contextmanager
def discard_graph_changes(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that enters a throwaway subtracer so nodes created inside the
    # block go to a separate graph and never reach the main graph. The subtracer is exited on
    # leaving the block even if an exception occurs.
    ctx = tx.output.subtracer("subgraph_wrapper", None)
    try:
        ctx.__enter__()
        yield
    finally:
        ctx.__exit__(None, None, None)


def check_meta_consistency_vt(
    vars1: list[VariableTracker],
    vars2: list[VariableTracker],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    # [FUNCTION DOC] Unwraps two lists of variable trackers into example values (fake tensor,
    # symbolic number, or python constant) and delegates to the runtime check_meta_consistency,
    # which raises if tensor metadata of the two lists differ. Any other variable kind triggers a
    # graph break. Returns whatever the delegate returns (none).
    from torch._higher_order_ops.utils import check_meta_consistency

    def _unwrap_var(var: VariableTracker) -> Any:
        # [FUNCTION DOC] HELPER: Returns the example value of a tensor, the sym number of a
        # SymNodeVariable, or the python constant; otherwise graph breaks.
        if var.is_tensor():
            # pyrefly: ignore[missing-attribute]
            return var.proxy.node.meta["example_value"]
        elif isinstance(var, SymNodeVariable):
            return var.sym_num
        elif var.is_python_constant():
            return var.as_python_constant()
        else:
            unimplemented(
                gb_type="cannot unwrap variable for check_meta_consistency",
                context=str(var),
                explanation=f"Expected {var} to be TensorVariable, SymNodeVariable, or ConstantVariable",
                hints=[],
            )

    unwrapped1 = [_unwrap_var(var) for var in vars1]
    unwrapped2 = [_unwrap_var(var) for var in vars2]

    return check_meta_consistency(
        unwrapped1,
        unwrapped2,
        lhs_name,
        rhs_name,
        include_contiguity=include_contiguity,
    )


@contextlib.contextmanager
def dynamo_enable_grad(
    tx: "InstructionTranslatorBase", enable: bool = True
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that sets the traced grad mode to the given flag via
    # GradModeVariable and restores the grad mode that was active on entry when the block exits
    # (also on exceptions). Side effect: emits grad-mode toggle state into the Dynamo output graph.
    from . import GradModeVariable

    org_value = torch.is_grad_enabled()
    try:
        GradModeVariable.create(tx, enable, initialized=True)
        yield
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)


@contextlib.contextmanager
def dynamo_allow_side_effects_in_hop(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that temporarily sets the current tracer's
    # allow_side_effects_in_hop flag to true and restores the previous value on exit.
    orig_val = tx.output.current_tracer.allow_side_effects_in_hop
    try:
        tx.output.current_tracer.allow_side_effects_in_hop = True
        yield
    finally:
        tx.output.current_tracer.allow_side_effects_in_hop = orig_val


def find_mismatched_vars(
    var: Any, types: type | tuple[type, ...], allow_none: bool = False
) -> set[VariableTracker]:
    # [FUNCTION DOC] Recursively walks python lists/tuples, list/tuple variables, dict variable
    # values and set variable elements, collecting leaf variables that are not instances of the
    # allowed types (none constants are exempt when allow_none is set). Returns the set of
    # offending leaves; no side effects.
    """
    Recursively finds variables whose type is not an instance of the specified types.
    Args:
        var: The variable to check.
        types: A tuple of allowed types.
        allow_none (bool): Whether to allow None values. Defaults to False.
    Returns:
        A set of variables whose type is not an instance of the specified types.
    """
    mismatched_vars = set()
    if isinstance(var, (list, tuple)):
        for item in var:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, (TupleVariable, ListVariable)):
        for item in var.items:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, ConstDictVariable):
        for value in var.items.values():
            mismatched_vars.update(find_mismatched_vars(value, types, allow_none))
    elif isinstance(var, SetVariable):
        for key in var.items:
            mismatched_vars.update(find_mismatched_vars(key.vt, types, allow_none))
    else:
        if not isinstance(var, types) and not (allow_none and var.is_constant_none()):
            mismatched_vars.add(var)
    return mismatched_vars


def only_consist_of(
    var: Any, types: tuple[type, ...], allow_none: bool = False
) -> bool:
    # [FUNCTION DOC] Returns true when every leaf of the nested variable structure is an instance
    # of the given types (optionally allowing none); implemented via find_mismatched_vars.
    mismatch_vars = find_mismatched_vars(var, types, allow_none=allow_none)
    return len(mismatch_vars) == 0


# A more read-able syntax sugar for creating a UserFunctionVariable for f
# and run call_function on it. Make it return a function to preserve the calling
# convention of the original f.
from torch._dynamo.utils import _make_inlined


def add_call_function(
    tx: "InstructionTranslatorBase",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    config: NestedCompileRegionOptions | None = None,
) -> VariableTracker:
    # [FUNCTION DOC] Emits a call_function node for fn with the given args/kwargs into the current
    # graph and wraps the resulting proxy in a variable tracker using flat_example_value. If a
    # nested compile region config is given, stores it in the node's custom metadata and raises
    # AssertionError unless the node target is invoke_subgraph. Returns the wrapped variable.
    from .builder import wrap_fx_proxy

    proxy = tx.output.create_proxy(
        "call_function",
        fn,
        args=args,
        kwargs=kwargs,
    )

    # Set backend metadata if provided
    if config is not None:
        if "custom" not in proxy.node.meta:
            # pyrefly: ignore [implicit-any]
            proxy.node.meta["custom"] = {}
        proxy.node.meta["custom"]["nested_region_config"] = config
        if proxy.node.target != torch._higher_order_ops.invoke_subgraph:
            raise AssertionError(
                "Expected proxy node target to be torch._higher_order_ops.invoke_subgraph"
            )

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=proxy,
        example_value=flat_example_value,
    )
    return flat_variable


def overwrite_tensor_vt_requires_grad(
    graph_output_vts: Iterable[VariableTracker], flat_variable: VariableTracker
) -> None:
    # [FUNCTION DOC] Pairs original output tensor variables with the HOP call's output variables
    # and copies requires_grad onto the originals, marking has_grad_fn when it is true. Needed
    # because grad mode is disabled inside autograd.Function. Raises AssertionError if a paired
    # subgraph output is not a TensorVariable. Mutates the original variables in place.
    # All outputs of autograd.Function have requires_grad=True. We turn off
    # grad_mode in autograd.Function, so our outputs naively have
    # requires_grad=False. So we hackily force them back on here. A better
    # solution would be to write python code that Dynamo could trace but we
    # decided that it wasn't worth it.
    # pyrefly: ignore[missing-attribute]
    for orig_vt, subgraph_vt in zip(graph_output_vts, flat_variable.items):
        if isinstance(orig_vt, variables.TensorVariable):
            if not isinstance(subgraph_vt, variables.TensorVariable):
                raise AssertionError("Expected subgraph_vt to be a TensorVariable")
            orig_vt.requires_grad = subgraph_vt.requires_grad
            if orig_vt.requires_grad:
                orig_vt.has_grad_fn = True


def overwrite_tensor_vt_proxy(
    graph_output_vts: Iterable[VariableTracker], flat_variable: VariableTracker
) -> None:
    # [FUNCTION DOC] Pairs each original graph-output variable with the corresponding variable
    # from the HOP call result and, for tensor, symbolic-number and custom-class-object
    # originals, replaces the original's proxy with the call result's proxy so later outer-graph
    # uses reference the HOP outputs. Raises AssertionError if the paired result is of an
    # unexpected kind. Mutates the original variables.
    # wrap_fx_proxy creates fresh variable trackers. However, the main program
    # after the speculate subgraph can still use the original tensor vts that
    # are still pointing to the nodes present in the subgraph. So, we reproxify
    # the original tensor vts with the subgraph outputs. This way, whenever the
    # outer graph uses an original vt, it uses the subgraph output.
    #
    # This is critical for maintaining the separation between:
    # - `body_r`: The output VT structure that Dynamo continues tracing (may
    #   contain non-proxyable objects, nested structures, etc.)
    # - `graph_output_vts`: Only the tensor/symint VTs that were actual graph
    #   outputs from speculate_subgraph
    #
    # By overwriting the proxies of VTs in `body_r` with the proxies from the
    # HOP call, we ensure the outer graph correctly references the HOP outputs
    # while still allowing `body_r` to contain arbitrary Python objects.
    # pyrefly: ignore[missing-attribute]
    for orig_vt, subgraph_vt in zip(graph_output_vts, flat_variable.items):
        if isinstance(
            orig_vt,
            (
                variables.SymNodeVariable,
                variables.TensorVariable,
                CustomClassObjectVariable,
            ),
        ):
            if not (
                subgraph_vt.is_tensor()
                or isinstance(subgraph_vt, (SymNodeVariable, CustomClassObjectVariable))
            ):
                raise AssertionError(
                    "Expected subgraph_vt to be a tensor, SymNodeVariable, or CustomClassObjectVariable"
                )
            orig_vt.proxy = subgraph_vt.proxy


def _call_function_with_auto_output_flattening(
    tx: "InstructionTranslatorBase",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    body_r: VariableTracker | None,
    graph_output_vts: VariableTracker | tuple[VariableTracker, ...],
    config: NestedCompileRegionOptions | None = None,
) -> VariableTracker | None:
    # [FUNCTION DOC] Emits the HOP call node via add_call_function and, when body_r is not none,
    # reproxifies the graph output variables to the call's outputs. Returns body_r unchanged
    # (or none) so tracing continues with the original output structure.
    """
    Create HOP call node and reproxify output VTs for HOPs with auto output semantics.

    This function is used by HOPs with auto output semantics (see speculate_subgraph_with_auto_output_flattening)
    to create the actual HOP call in the FX graph and properly handle the output variable trackers.

    The key operation is "reproxifying" - updating the proxies of the original tensor VTs
    (from body_r) to point to the HOP call outputs, ensuring the outer graph correctly
    references the HOP outputs while allowing body_r to contain arbitrary Python objects.

    Args:
        tx: The instruction translator
        fn: The HOP function to call
        args: Arguments for the HOP call (typically includes the subgraph node)
        kwargs: Keyword arguments for the HOP call
        flat_example_value: Example value for the HOP output
        body_r: The output VT structure that Dynamo continues tracing with (may be None)
        graph_output_vts: Tensor/symint VTs that were actual graph outputs

    Returns:
        The body_r VT (unchanged), which Dynamo will continue tracing with
    """

    flat_variable = add_call_function(tx, fn, args, kwargs, flat_example_value, config)
    if body_r is not None:
        # pyrefly: ignore[bad-argument-type]
        overwrite_tensor_vt_proxy(graph_output_vts, flat_variable)
    return body_r


def _call_function_and_unflatten_output(
    tx: "InstructionTranslatorBase",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    ret_spec: OutputSpec,
    body_r: VariableTracker | None,
) -> VariableTracker:
    # [FUNCTION DOC] Emits the HOP call node, wraps its flat result, reproxifies the tensor/symint
    # variables of body_r to the call outputs, drops trailing extra intermediate outputs, re-inserts
    # constant values removed during flattening, converts the flat tuple to a list, and unflattens
    # it with the saved treespec (or returns the flat variable when no treespec). Returns the
    # reconstructed output variable. Raises AssertionError on mismatched kinds of paired outputs.
    from .builder import SourcelessBuilder, wrap_fx_proxy

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=tx.output.create_proxy(
            "call_function",
            fn,
            args=args,
            kwargs=kwargs,
        ),
        example_value=flat_example_value,
    )

    # wrap_fx_proxy creates fresh variable trackers. However, the main program
    # after the speculate subgraph can still use the original tensor vts that
    # are still pointing to the nodes present in the subgraph. So, we reproxify
    # the original tensor vts with the subgraph outputs. This way, whenever the
    # outer graph uses an original vt, it uses the subgraph output.
    if body_r is not None:
        # mypy: ignore[attr-defined]
        for orig_vt, subgraph_vt in zip(body_r.items, flat_variable.items):
            if orig_vt.is_tensor() or isinstance(orig_vt, SymNodeVariable):
                if not (
                    subgraph_vt.is_tensor() or isinstance(subgraph_vt, SymNodeVariable)
                ):
                    raise AssertionError(
                        "Expected subgraph_vt to be a tensor or SymNodeVariable"
                    )
                orig_vt.proxy = subgraph_vt.proxy

    if ret_spec.num_intermediate_nodes_as_outputs:
        # The treespec was computed w/o any extra intermediate outputs. At this
        # point, it is safe to just get rid of the extra outputs
        flat_variable = SourcelessBuilder.create(
            tx,
            flat_variable.items[  # mypy: ignore[attr-defined]
                : -ret_spec.num_intermediate_nodes_as_outputs
            ],
        )

    if ret_spec.masks_to_filter_const_values:
        from torch._dynamo.external_utils import insert_const_values_with_mask

        # During flattening, we removed the constant values. To ensure Dynamo
        # can trace correctly, insert back the constant values in the output.
        flat_variable = _make_inlined(tx, insert_const_values_with_mask)(
            flat_variable, ret_spec.masks_to_filter_const_values, ret_spec.const_values
        )

    # Transform variable back into a list (previously made into a tuple by
    # speculate_subgraph function) so as to respect the pytree API typing.
    flat_list_variable = SourcelessBuilder.create(tx, list).call_function(
        tx, [flat_variable], {}
    )
    return (
        _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, ret_spec.treespec)
        if ret_spec.treespec
        else flat_variable
    )


def _assert_tensors_nonaliasing(inputs: Any, outputs: Any) -> None:
    # [FUNCTION DOC] HELPER: Flattens inputs and outputs and raises AssertionError if any tensor
    # object (by identity) appears in both.
    input_tensor_ids = {
        id(t) for t in pytree.tree_leaves(inputs) if isinstance(t, torch.Tensor)
    }
    output_tensor_ids = {
        id(t) for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
    }
    if not input_tensor_ids.isdisjoint(output_tensor_ids):
        raise AssertionError("inputs to function body cannot alias outputs")


def get_tensor_storages(tensor: torch.Tensor) -> set[StorageWeakRef]:
    # [FUNCTION DOC] Returns a set holding a weak reference to the tensor's typed storage (empty
    # set for non-tensors). Raises NotImplementedError for sparse/sparse-CSR tensors and for
    # traceable wrapper subclasses.
    """
    Get storage references from a tensor.

    Handles regular tensors. Raises NotImplementedError for sparse tensors
    and traceable wrapper subclasses.

    Args:
        tensor: The tensor to extract storages from

    Returns:
        Set of StorageWeakRef objects for the tensor's storage(s)
    """
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    storages: set[StorageWeakRef] = set()

    if not isinstance(tensor, torch.Tensor):
        return storages

    if tensor.is_sparse or tensor.is_sparse_csr:
        raise NotImplementedError("get_tensor_storages does not support sparse tensors")

    if is_traceable_wrapper_subclass(tensor):
        raise NotImplementedError(
            "get_tensor_storages does not support traceable wrapper subclasses"
        )
    else:
        storages.add(StorageWeakRef(tensor._typed_storage()))

    return storages


def subgraph_mutated_input_storages(
    subgraph: torch.fx.Graph, mutated_input_indices: Iterable[int]
) -> set[StorageWeakRef]:
    # [FUNCTION DOC] Looks up each mutated placeholder index among the subgraph's placeholder
    # nodes, reads its example value and unions the tensor storages. Raises AssertionError if an
    # index is out of range or the example value is not a tensor. Returns the storage set.
    """Map subgraph placeholder indices flagged as mutated to their tensor storages.

    Used by HigherOrderVariable subclasses to translate
    ``_dynamo_mutated_input_indices`` into a parent-side storage set,
    which is then matched against the parent's full input list to recover
    the parent-side mutated index list passed as a HOP kwarg.
    """
    placeholders = [node for node in subgraph.nodes if node.op == "placeholder"]
    storages: set[StorageWeakRef] = set()
    for idx in mutated_input_indices:
        if idx >= len(placeholders):
            raise AssertionError(
                f"mutated input index {idx} out of range for {len(placeholders)} placeholders"
            )
        placeholder = placeholders[idx]
        example_value = placeholder.meta.get(
            "example_value", placeholder.meta.get("val", None)
        )
        if not isinstance(example_value, torch.Tensor):
            raise AssertionError("The mutated input must be a tensor")
        storages |= get_tensor_storages(example_value)
    return storages


def parent_mutated_input_indices(
    parent_inputs: Sequence["VariableTracker | Proxy"],
    mutated_input_storages: set[StorageWeakRef],
) -> list[int]:
    # [FUNCTION DOC] Returns indices of parent inputs (variables or proxies) whose tensor storage
    # is in the mutated storage set; returns an empty list immediately when the set is empty.
    # Raises AssertionError if a non-variable input is not a Proxy.
    """Walk a HOP's full parent-side input list and return the indices whose
    tensor storage matches one of the storages mutated inside any of the
    subgraphs.

    If no subgraph mutation was detected, returns the empty list without
    inspecting any storages. Some inputs (e.g. BatchedTensor under vmap)
    deliberately refuse storage access; we must not trip on those when
    there's nothing to match against.
    """
    if not mutated_input_storages:
        return []
    mutated: list[int] = []
    for idx, inp in enumerate(parent_inputs):
        example_value = None
        if isinstance(inp, VariableTracker):
            if inp.is_tensor():
                example_value = inp.as_proxy().node.meta.get("example_value", None)
        else:
            if not isinstance(inp, Proxy):
                raise AssertionError("Expected inp to be a Proxy")
            example_value = inp.node.meta.get("example_value", inp.node.meta.get("val"))

        if isinstance(example_value, torch.Tensor):
            for storage in get_tensor_storages(example_value):
                if storage in mutated_input_storages:
                    mutated.append(idx)
                    break
    return mutated


class StorageAliasingTracker:
    # [CLASS DOC] Holds excluded_storages, a set of weak storage references seen in graph inputs
    # and chosen outputs. Methods collect storages from inputs/outputs and check_and_track decides
    # whether a candidate intermediate aliases anything already tracked (recording it if not).
    """
    Tracks storage references to detect aliasing between tensors.

    This class encapsulates the logic for collecting storages from tensors
    and checking for aliasing conflicts. Used to filter intermediate outputs
    that would create input-output or output-output aliasing.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the tracker with an empty excluded storage set.
        self.excluded_storages: set[StorageWeakRef] = set()

    def _collect_storages_from_tensor(self, example_value: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Adds the tensor's storages to the excluded set.
        self.excluded_storages.update(get_tensor_storages(example_value))

    def collect_from_inputs(self, tx: "InstructionTranslatorBase") -> None:
        # [FUNCTION DOC] Walks the leading placeholder nodes of the current graph (stops at the
        # first non-placeholder) and records the storages of their fake tensor values.
        """Collect storages from graph input placeholders."""
        from torch._higher_order_ops.utils import _collect_fake_inputs

        for node in tx.output.graph.nodes:
            if node.op == "placeholder":
                example_value = _collect_fake_inputs([node])[0]
                if isinstance(example_value, torch.Tensor):
                    self._collect_storages_from_tensor(example_value)
            else:
                break

    def collect_from_outputs(self, graph_output_vts: list[VariableTracker]) -> None:
        # [FUNCTION DOC] Records the storages of each output variable's proxy example value.
        """Collect storages from existing graph outputs."""
        from torch._higher_order_ops.utils import _collect_fake_inputs

        for vt in graph_output_vts:
            proxy = vt.as_proxy()
            example_value = _collect_fake_inputs([proxy.node])[0]
            if isinstance(example_value, torch.Tensor):
                self._collect_storages_from_tensor(example_value)

    def check_and_track(self, proxy_node: Proxy) -> bool:
        # [FUNCTION DOC] Returns true for non-tensor values or tensors whose storage does not
        # overlap the excluded set; in the latter case also records the tensor's storage (skipped
        # for sparse tensors and wrapper subclasses). Returns false on aliasing.
        """
        Check if a tensor can be added as a subgraph output without causing aliasing issues.

        Given a proxy node, extracts its example tensor value and checks if its storage
        aliases with any previously tracked storages (from inputs or other outputs).
        If there's no aliasing conflict, the tensor's storage is added to the tracked set.

        Args:
            proxy_node: An FX proxy node whose example_value is the tensor to check.

        Returns:
            True if the tensor doesn't alias with tracked storages (safe to add as output),
            False if it aliases (should be filtered out).
        """
        from torch._higher_order_ops.utils import _collect_fake_inputs
        from torch.multiprocessing.reductions import StorageWeakRef
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        example_value = _collect_fake_inputs([proxy_node])[0]

        # Non-tensor outputs (e.g., symints) don't have aliasing concerns
        if not isinstance(example_value, torch.Tensor):
            return True

        # Check if any storage aliases with existing inputs/outputs
        tensor_storages = get_tensor_storages(example_value)
        if tensor_storages & self.excluded_storages:
            return False

        # Track this tensor's storage (for wrapper subclasses, inner storages were already checked)
        if not is_traceable_wrapper_subclass(example_value):
            if not (example_value.is_sparse or example_value.is_sparse_csr):
                self.excluded_storages.add(
                    StorageWeakRef(example_value._typed_storage())
                )

        return True


def taint_filtered_vt(vt: VariableTracker) -> None:
    # [FUNCTION DOC] Replaces the variable's as_proxy method so that using it later raises a
    # RuntimeError explaining the intermediate was dropped from subgraph outputs due to aliasing
    # and should be cloned. Mutates the variable instance.
    """Mark a VT as filtered due to aliasing so it raises a clear error if used."""
    original_as_proxy = vt.as_proxy

    def tainted_as_proxy() -> Any:
        # [FUNCTION DOC] HELPER: Resolves the original proxy for naming, then always raises
        # RuntimeError describing the aliasing problem.
        proxy = original_as_proxy()
        raise RuntimeError(
            f"An intermediate tensor '{proxy.node.name}' created inside a "
            f"higher-order op subgraph aliases an input or output, so it was "
            f"not included in the subgraph outputs. However, it is being used "
            f"in the outer graph (e.g., via a side effect like list.append). "
            f"To fix this, clone the tensor before capturing it "
            f"(e.g., use tensor.clone() instead of tensor)."
        )

    vt.as_proxy = tainted_as_proxy  # type: ignore[method-assign]


def collect_intermediate_outputs(
    tx: "InstructionTranslatorBase",
    subtracer: "SubgraphTracer",
    graph_output_vts: list[VariableTracker],
    filter_aliased_intermediates: bool = False,
) -> list[VariableTracker]:
    # [FUNCTION DOC] Returns tracked proxyable variables of the subtracer that are not already
    # graph outputs, skipping float symbolic numbers. When filtering is enabled, intermediates
    # aliasing graph inputs or other outputs are excluded (and tainted so later use raises).
    # Raises AssertionError on an internal inconsistency.
    extra_outputs = []
    existing_out_proxies = {vt.as_proxy() for vt in graph_output_vts}

    # Build the aliasing tracker if we're filtering
    tracker = None
    if filter_aliased_intermediates:
        tracker = StorageAliasingTracker()
        tracker.collect_from_inputs(tx)
        tracker.collect_from_outputs(graph_output_vts)

    for out in subtracer.tracked_proxyable_vt:
        proxy = out.as_proxy()

        # Skip if already in output
        if proxy in existing_out_proxies:
            continue

        # TODO floats are not supported in HOP input/output
        if isinstance(out, SymNodeVariable) and out.python_type() is float:
            continue

        if not filter_aliased_intermediates:
            extra_outputs.append(out)
        else:
            # Filter out intermediates that alias with inputs or outputs.
            # This is needed for HOPs like invoke_subgraph that don't support aliasing.
            if tracker is None:
                raise AssertionError(
                    "tracker must not be None for intermediate filtering"
                )
            if tracker.check_and_track(proxy.node):
                extra_outputs.append(out)
            else:
                taint_filtered_vt(out)

    return extra_outputs


def _check_all_tensorvariable(args: list[VariableTracker]) -> None:
    # [FUNCTION DOC] HELPER: Realizes each arg and graph breaks unless all are exactly
    # TensorVariable.
    if not all(type(a.realize()) is TensorVariable for a in args):
        unimplemented(
            gb_type="HOP: non torch.Tensor leaf",
            context=f"args types: {[type(a.realize()) for a in args]}",
            explanation="Expected all leaves to be of torch.Tensor type.",
            hints=[],
        )


def _check_supported_callable_arg(
    tx: "InstructionTranslatorBase", func_var: VariableTracker, arg_name: str
) -> None:
    # [FUNCTION DOC] HELPER: Evaluates the callable builtin on the variable during tracing and
    # graph breaks if the variable is not callable.
    from .builder import SourcelessBuilder

    is_callable = (
        SourcelessBuilder.create(tx, callable)
        .call_function(tx, [func_var], {})
        .as_python_constant()
    )
    if not is_callable:
        unimplemented(
            gb_type="HOP: non-callable variable",
            context=f"arg name: {arg_name}, func_var type: {str(func_var)}",
            explanation=f"{arg_name} should be a callable but is of type {str(func_var)}.",
            hints=[],
        )


def _call_while_loop(
    self: Union[
        "WhileLoopHigherOrderVariable", "WhileLoopStackOutputHigherOrderVariable"
    ],
    tx: "InstructionTranslatorBase",
    args: list[VariableTracker],
    kwargs: dict[str, VariableTracker],
    stack_output: bool,
    hop_name: str,
) -> VariableTracker:
    # [FUNCTION DOC] HELPER: Shared tracing for while_loop and its stack-output variant. Validates
    # four positional args (cond_fn, body_fn, operands, additional_inputs; keyword forms moved
    # to positional), graph breaks on bad usage. Clones/unspecializes carried inputs under
    # discarded graph changes, speculates cond and body subgraphs with manual flat inputs, checks
    # cond returns a bool scalar tensor (a constant true graph-breaks as infinite loop, constant
    # false returns operands unchanged), checks body output meta matches the carries, merges lifted
    # free variables, records mutated input indices, installs both subgraphs and emits the HOP call
    # (with mutated_arg_indices only when not stack_output). Returns the unflattened output.
    from torch._higher_order_ops.while_loop import _create_unbacked_symint

    args, kwargs = LazyVariableTracker.realize_all((args, kwargs))
    cond_fn, body_fn, operands, additional_inputs = args

    # Input checks
    for i, k in enumerate(["cond_fn", "body_fn", "operands"]):
        if v := kwargs.pop(k, None):
            if i != len(args):
                raise AssertionError(
                    "did not provide the right number of non-keyword args"
                )
            args.append(v)

    if kwargs or len(args) != 4:
        unimplemented(
            gb_type="torch.while_loop: improper args/kwargs",
            context=f"args: {args}, kwargs: {kwargs}",
            explanation=f"torch.while_loop expects 4 positional arguments (got {len(args)}) "
            f"and no keyword arguments (got {len(kwargs)}) "
            "Usage: while_loop(cond_fn, body_fn, operands)",
            hints=[
                *graph_break_hints.USER_ERROR,
            ],
        )

    # cond_fn and body_fn input check
    _check_supported_callable_arg(tx, cond_fn, "cond_fn")
    _check_supported_callable_arg(tx, body_fn, "body_fn")

    # operands input check
    operands_seq = unpack_iterable(tx, operands)

    # additional_inputs input check
    if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
        unimplemented(
            gb_type="torch.while_loop: improper additional_inputs",
            context=str(additional_inputs),
            explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )
    additional_inputs_seq = unpack_iterable(tx, additional_inputs)

    with discard_graph_changes(tx):
        # Note: this must be run under discard graph changes.
        def unspecialize_carried_inputs(
            tx: "InstructionTranslatorBase", carry: VariableTracker
        ) -> VariableTracker:
            # [FUNCTION DOC] HELPER: Replaces an int constant or symbolic number carry with a fresh
            # unbacked symint graph input; otherwise asserts the carry is a tensor and returns a
            # clone with its constant marker cleared. Raises AssertionError for other kinds.
            # See NOTE [unspecialize int carry with unbacked symints]
            if (
                carry.is_python_constant()
                and isinstance(carry.as_python_constant(), int)
            ) or isinstance(carry, SymNodeVariable):
                example_value = _create_unbacked_symint(
                    tx.output.fake_mode, ignore_fresh_unbacked_symbols=True
                )
                proxy = tx.output.current_tracer.create_graph_input(
                    "unbacked_symint", type(example_value), example_value
                )
                return SymNodeVariable.create(tx, proxy, example_value)
            else:
                # See NOTE [unspecialize constant tensor carry]
                if not carry.is_tensor():
                    raise AssertionError("Expected carry to be a tensor")
                cloned_carry = carry.clone()
                # type: ignore[attr-defined]
                cloned_carry.proxy.node.meta["example_value"].constant = None
                return cloned_carry

        # clone inputs across subgraphs, to avoid unbacked memoization in fake prop
        cond_operands_seq = [
            unspecialize_carried_inputs(
                tx,
                (
                    carry.call_method(tx, "clone", args=[], kwargs={})
                    if carry.is_tensor()
                    else carry
                ),
            )
            for carry in operands_seq
        ]
        body_operands_seq = [
            unspecialize_carried_inputs(
                tx,
                (
                    carry.call_method(tx, "clone", args=[], kwargs={})
                    if carry.is_tensor()
                    else carry
                ),
            )
            for carry in operands_seq
        ]

    # create cond subgrpahs
    (
        (cond_r, _cond_treespec),
        cond_graph,
        cond_lifted_freevars,
    ) = speculate_subgraph(
        tx,
        cond_fn,
        cond_operands_seq + additional_inputs_seq,
        {},
        hop_name,
        source_target=self.value,
        # NOTE [why we cannot use "automatic" for while_loop]:
        # The reason is that we want to enforce
        # the ordering of inputs and outputs to be consistent and the ordering
        # of cond_fn and body_fn to the consistent.
        # e.g. suppose we use "automatic" and we have:
        #
        # def body_fn(ph1, ph2):
        #   new_a, new_b = ph2.cos(), ph1.sin()
        #   return new_a, new_b
        #
        # a, b = torch.randn(3), torch.randn(3)
        # new_a, new_b = body_fn(a, b)
        #
        # Using automatic, the ordering of arguments will be the order that they're
        # used. In this example, the capture graph looks like:
        #
        # def captured_body(ph1, ph2):
        #   new_a, new_b = ph1.cos(), ph2.add_(1)
        #   return new_a, new_b
        #
        # This is fine when we change the calling convention of captured_body to be
        # new_a, new_b = captured_body(b, a).
        # But for while_loop, the next iteration's input is previous iteration output
        # we'll end up feeding captured_body(new_a, new_b) instead.
        # So it's best we always enforce the ordering of carried_inputs the same as outputs
        # with "flatten_manual".
        set_subgraph_inputs="flatten_manual",
        supports_input_mutation=self.supports_input_mutation,
        supports_aliasing=self.supports_aliasing,
        remove_consts_from_outputs=False,
    )
    cond_nn_modules = dict(tx.output.nn_modules)
    validate_subgraph_output_types(cond_r)
    if cond_r.is_tensor():
        cond_r_meta = _extract_tensor_metadata(
            # type: ignore[attr-defined]
            cond_r.proxy.node.meta["example_value"],
            include_contiguity=False,
        )
        if cond_r_meta.dtype != torch.bool or cond_r_meta.shape != torch.Size([]):
            unimplemented(
                gb_type="torch.while_loop: unsupported cond_fn return type",
                context=str(cond_r),
                explanation=f"Expected cond_fn to return a scalar tensor or a bool but got {cond_r_meta.shape}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
    elif cond_r.is_python_constant():
        # short-circuiting while_loop when cond_fn returns a constant such as 0, 1 True or False
        pred = cond_r.as_python_constant()
        if pred:
            unimplemented(
                gb_type="torch.while_loop: infinite loop detected",
                context=str(cond_r),
                explanation=f"Infinite loop detected because while_loop's cond_fn always returns the same value {pred}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
        else:
            return operands

    # create body subgraph
    (
        (body_r, body_treespec),
        body_graph,
        body_lifted_freevars,
    ) = speculate_subgraph(
        tx,
        body_fn,
        body_operands_seq + additional_inputs_seq,
        {},
        hop_name,
        source_target=self.value,
        set_subgraph_inputs="flatten_manual",
        should_flatten_outputs=True,
        supports_input_mutation=self.supports_input_mutation,
        supports_aliasing=self.supports_aliasing,
        remove_consts_from_outputs=False,
    )
    validate_subgraph_output_types(body_r)
    cond_mutated_inputs = set(getattr(cond_graph, "_dynamo_mutated_input_indices", ()))
    body_mutated_inputs = set(getattr(body_graph, "_dynamo_mutated_input_indices", ()))

    mutated_input_storages = subgraph_mutated_input_storages(
        cond_graph, cond_mutated_inputs
    ) | subgraph_mutated_input_storages(body_graph, body_mutated_inputs)

    # We set include contiguity=False because we have vmap x HOP tests, where if
    # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
    # "querying is_contiguous inside of vmap for memory_format other than
    # torch.contiguous_format is not yet implemented". This is okay because stride
    # is still checked.
    check_meta_consistency_vt(
        unpack_iterable(tx, body_r),
        operands_seq,
        "body_fn_output",
        "carried_inputs",
        include_contiguity=False,
    )

    shared, (cond_unique, body_unique) = _merge_graph_inputs(
        [cond_graph, body_graph],
        [cond_lifted_freevars, body_lifted_freevars],
        ["cond_fn", "body_fn"],
    )

    additional_lifted_inputs = shared + cond_unique + body_unique
    all_while_loop_inputs: list[VariableTracker | Proxy] = (
        list(operands_seq)
        + list(additional_inputs_seq)
        + list(additional_lifted_inputs)
    )
    mutated_inputs = parent_mutated_input_indices(
        all_while_loop_inputs, mutated_input_storages
    )

    mutated_arg_indices = ",".join(str(i) for i in mutated_inputs)

    body_nn_modules = dict(tx.output.nn_modules)

    cond_gm = torch.fx.GraphModule(cond_nn_modules, cond_graph)
    body_gm = torch.fx.GraphModule(body_nn_modules, body_graph)
    cond_name = tx.output.install_subgraph("cond_fn", cond_gm)
    body_name = tx.output.install_subgraph("body_fn", body_gm)

    cond_node = make_attr(tx, cond_name)
    body_node = make_attr(tx, body_name)

    operands_proxy = tuple(operand.as_proxy() for operand in operands_seq)
    additional_inputs_proxy = tuple(
        [inp.as_proxy() for inp in additional_inputs_seq] + additional_lifted_inputs
    )
    p_args = (
        cond_node,
        body_node,
        operands_proxy,
        additional_inputs_proxy,
    )
    hop_kwargs = (
        {"mutated_arg_indices": mutated_arg_indices}
        if not stack_output and mutated_arg_indices
        else {}
    )
    return _call_function_and_unflatten_output(
        tx,
        self.value,
        p_args,
        hop_kwargs,
        None,
        body_treespec,
        body_r,
    )


def are_same_graph_modules(
    fn_name: str, a_mod: GraphModule, b_mod: GraphModule, fake_mode: "FakeTensorMode"
) -> bool:
    # [FUNCTION DOC] Structurally compares two graph modules node by node: same ops, placeholder
    # fake tensor metadata / symint identity, call_function and call_method targets and flattened
    # args (through a node correspondence map), output args, and recursively nested graph module
    # attributes. Returns true if equivalent, false otherwise (logging the first mismatch at debug
    # level). Raises NotImplementedError for non-graph-module get_attr or call_module nodes.
    from torch._subclasses._fake_tensor_utils import _CacheKeyState
    from torch._subclasses.fake_tensor import extract_tensor_metadata

    # Maps the equivalent nodes from a to b
    # pyrefly: ignore [implicit-any]
    node_map = {}

    def check_all_args(a_nodes: Iterable[Any], b_nodes: Iterable[Any]) -> bool:
        # [FUNCTION DOC] HELPER: Pairwise compares flattened args: nodes must map via node_map to
        # the other side, slices are compared by their parts, other values by equality.
        for arg_a, arg_b in zip(a_nodes, b_nodes):
            if isinstance(arg_a, torch.fx.Node):
                if node_map[arg_a] != arg_b:
                    return False
            elif isinstance(arg_a, slice):
                if not isinstance(arg_b, slice):
                    return False
                if not check_all_args(
                    (arg_a.start, arg_a.stop, arg_a.step),
                    (arg_b.start, arg_b.stop, arg_b.step),
                ):
                    return False
            elif arg_a != arg_b:
                # This is a catch-all for everything else. `slice` was a
                # surprise but can there be other data structures that can
                # contain fx.Nodes in them?
                return False
        return True

    for a_node, b_node in zip(a_mod.graph.nodes, b_mod.graph.nodes):
        if a_node.op != b_node.op:
            return False

        if a_node.op == "placeholder":
            a_value = a_node.meta["example_value"]
            b_value = b_node.meta["example_value"]

            if isinstance(a_value, torch.Tensor):
                if not isinstance(b_value, torch.Tensor):
                    return False
                # Extract fake tensor metadata for a and b and then compare
                # pyrefly: ignore [implicit-any]
                a_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                a_metadata = extract_tensor_metadata(a_value)
                a_metadata._flatten_into(a_result, fake_mode, state)

                b_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                b_metadata = extract_tensor_metadata(b_value)
                b_metadata._flatten_into(b_result, fake_mode, state)
                if a_result != b_result:
                    return False
            elif isinstance(a_value, torch.SymInt):
                if not isinstance(b_value, torch.SymInt):
                    return False
                if a_value is not b_value:
                    return False
        elif a_node.op == "call_function":
            if a_node.target is not b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug(
                    "%s: Graph comparison failed at node (call_function): %s",
                    fn_name,
                    a_node,
                )
                return False
        elif a_node.op == "call_method":
            if a_node.target != b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug(
                    "%s: Graph comparison failed at node (call_method) : %s",
                    fn_name,
                    a_node,
                )
                return False
        elif a_node.op == "output":
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug("%s: Graph comparison failed at the output node", fn_name)
                return False
        elif a_node.op == "get_attr":
            a_attr = getattr(a_mod, a_node.target)
            b_attr = getattr(b_mod, b_node.target)
            if isinstance(a_attr, torch.fx.GraphModule):
                if not isinstance(b_attr, torch.fx.GraphModule):
                    return False
                # This is an example of a HOP inside a HOP
                if not are_same_graph_modules(fn_name, a_attr, b_attr, fake_mode):
                    return False
            else:
                # TODO - write an example with tensor as a graph attribute in
                # the Fx graph
                raise NotImplementedError(f"get_attr with {type(a_attr)}")
        else:
            # TODO - call_module is not supported because Dynamo Fx graph does
            # not install a call_module
            raise NotImplementedError(f"Graph equivalence check saw a {a_node.op}")

        # Two nodes are equal - add them to them map
        node_map[a_node] = b_node

    return True


def validate_args_and_maybe_create_graph_inputs(
    sub_args: list[VariableTracker],
    tracer: "SubgraphTracer",
    tx: "InstructionTranslatorBase",
    set_subgraph_inputs: SetSubgraphInputs,
    description: str,
    sub_args_names: Sequence[str] | None = None,
) -> list[Any]:
    # [FUNCTION DOC] Turns sub_args into the arguments used to trace a subgraph according to
    # set_subgraph_inputs. "automatic": returned unchanged. "automatic_with_forced_inputs":
    # every tensor variable gets a new graph placeholder. "manual": constants get an unused
    # placeholder, autograd function contexts and graph-placeable values get placeholders
    # (rewrapped), anything else graph breaks. "flatten_manual": pytree-flatten, handle as
    # manual, then unflatten. Mutates the tracer by creating graph inputs. Raises AssertionError
    # on missing parent tracer, too few names or non-variable args.
    from . import AutogradFunctionContextVariable
    from .builder import SourcelessBuilder, wrap_fx_proxy_cls

    if tracer.parent is None:
        raise AssertionError("tracer.parent must not be None")

    if set_subgraph_inputs == "flatten_manual":
        flat_args, tree_spec = unpack_iterable(
            tx,
            _make_inlined(tx, pytree.tree_flatten)(
                SourcelessBuilder.create(tx, sub_args)
            ),
        )

        flat_inputs = validate_args_and_maybe_create_graph_inputs(
            unpack_iterable(tx, flat_args),
            tracer,
            tx,
            set_subgraph_inputs="manual",
            description=description,
        )

        return unpack_iterable(
            tx,
            _make_inlined(tx, pytree.tree_unflatten)(
                SourcelessBuilder.create(tx, list(flat_inputs)), tree_spec
            ),
        )
    else:
        if sub_args_names is not None:
            # Can be greater if user passes some args as kwargs
            if len(sub_args_names) < len(sub_args):
                raise AssertionError(
                    "sub_args_names must have at least as many entries as sub_args"
                )
        args = []
        for idx, a in enumerate(sub_args):
            if not isinstance(a, VariableTracker):
                raise AssertionError("Expected each sub_arg to be a VariableTracker")
            new_arg = None
            if set_subgraph_inputs == "automatic":
                args.append(a)
                continue
            elif set_subgraph_inputs == "automatic_with_forced_inputs":
                if isinstance(a, LazyVariableTracker):
                    a = a.realize()
                if isinstance(a, variables.TensorVariable):
                    node = a.maybe_fx_node()
                    if node is None:
                        raise AssertionError(
                            "Expected TensorVariable to have an fx node"
                        )
                    example_value = node.meta["example_value"]
                    arg_name = (
                        a.as_proxy().node.name
                        if sub_args_names is None
                        else sub_args_names[idx]
                    )
                    new_proxy = tracer.create_graph_input(
                        arg_name, a.python_type(), example_value
                    )
                    example_value = node.meta.get("example_value", None)
                    a = wrap_fx_proxy_cls(
                        target_cls=type(a),
                        tx=tx,
                        proxy=new_proxy,
                        example_value=example_value,
                    )
                args.append(a)
                continue

            if a.is_python_constant():
                # This arg is not used in the body of the higher order op.
                # Currently, this new input is added to make the calls
                # happy, which expect a fixed number of arguments. In
                # future, we can clean this up.
                arg_name = (
                    "const_unused"
                    if sub_args_names is None
                    else f"const_unused_{sub_args_names[idx]}"
                )
                tracer.create_graph_input(
                    arg_name, a.python_type(), a.as_python_constant()
                )
                new_arg = a
            # Weird special case, we probably want to delete it or fold it
            # into the next case (of `a` being placeable into a graph)
            elif isinstance(a, AutogradFunctionContextVariable):
                example_value = a.as_proxy().node.meta["example_value"]
                arg_name = (
                    a.as_proxy().node.name
                    if sub_args_names is None
                    else sub_args_names[idx]
                )
                tracer.create_graph_input(arg_name, a.python_type(), example_value)
                new_arg = a
            # If `a` can be put into a graph
            elif a.maybe_fx_node() is not None:
                node = a.maybe_fx_node()
                if node is None:
                    raise AssertionError(
                        "Expected a non-None fx node after maybe_fx_node() check"
                    )
                example_value = node.meta.get("example_value", None)
                arg_name = node.name if sub_args_names is None else sub_args_names[idx]
                new_proxy = tracer.create_graph_input(
                    arg_name, a.python_type(), example_value
                )
                new_arg = wrap_fx_proxy_cls(
                    target_cls=type(a),
                    tx=tx,
                    proxy=new_proxy,
                    example_value=example_value,
                )
            # If `a` cannot be put into a graph
            else:
                # HOPs work much better if they use speculate_subgraph(set_subgraph_inputs="automatic").
                unimplemented(
                    gb_type="HOP body taking non-Tensor as input",
                    context=str(sub_args),
                    explanation=f"{description} with body that accepts non-Tensors as input. "
                    f"Got type {a.python_type()} at index {idx}.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            args.append(new_arg)
        return args


# N-branch generalization of former _merge_graph_inputs. Used by torch.cond (2 branches)
# and torch.switch (N branches): both must produce subgraphs with a unified
# placeholder signature so the higher order op can pass a single operand list.
#
# Returns (shared, unique_per_branch):
#   * shared: proxies present in every branch's lifted freevars. For plain
#     proxies, "present" means by identity. For get_attr proxies, "present"
#     means by node target (branches in separate tracing contexts can produce
#     distinct proxies for the same nn module attr); the proxy from the first
#     branch is chosen as the canonical representative.
#   * unique_per_branch[i]: proxies lifted by branch i, not shared, and not
#     already assigned to an earlier branch's unique block. With N > 2 a
#     freevar can be lifted by some but not all branches; the first such branch
#     claims it, and later branches that also lift it reuse the same placeholder.
#     For N == 2 any non-shared freevar is lifted by exactly one branch -> no-op.
# Each block is sorted by node name for determinism.
#
# Side effect: each graph is rewritten in-place so its placeholders follow the
# 1 + N block layout: shared block (no suffix), then one block per branch
# (suffixed with "_<branch_name>") holding that branch's unique freevars. A
# branch graph wires its inner uses to whichever block holds the placeholder
# for the freevar it lifted (its own block, or the earlier branch that
# claimed it); the remaining blocks become unused placeholders for signature
# alignment.
def _merge_graph_inputs(
    graphs: list[torch.fx.Graph],
    lifted_freevars: list[dict[Proxy, Proxy]],
    names: list[str],
) -> tuple[list[Proxy], list[list[Proxy]]]:
    # [FUNCTION DOC] HELPER: See the comment above. Computes the shared and per-branch unique
    # lifted free variable proxies (deduplicating get_attr proxies by target, sorted by node
    # name), then rewrites every graph's placeholders in place to the unified shared-plus-
    # per-branch layout. Returns (shared, unique_per_branch). Raises AssertionError on length
    # mismatch or zero graphs.
    n = len(graphs)
    if n != len(lifted_freevars) or n != len(names):
        raise AssertionError(
            "Expected graphs, lifted_freevars and names to have the same length"
        )
    if n == 0:
        raise AssertionError("Expected at least one graph")

    # Step 1: pick a canonical proxy for every distinct outer freevar across
    # branches, deduplicating get_attrs that share a target. A canonical proxy
    # is shared iff every branch contributed to it.
    get_attr_canonical: dict[Any, Proxy] = {}
    per_branch_outer_to_canonical: list[dict[Proxy, Proxy]] = []
    canonical_branch_count: Counter[Proxy] = Counter()
    for branch_lifted in lifted_freevars:
        outer_to_canonical: dict[Proxy, Proxy] = {}
        for outer in branch_lifted:
            canonical = outer
            if outer.node.op == "get_attr":
                target = outer.node.target
                canonical = get_attr_canonical.setdefault(target, outer)
            outer_to_canonical[outer] = canonical
        per_branch_outer_to_canonical.append(outer_to_canonical)
        canonical_branch_count.update(set(outer_to_canonical.values()))

    shared_canonical = {c for c, count in canonical_branch_count.items() if count == n}

    # Step 2: derive the shared block and the per-branch unique blocks. Plain
    # proxies are the same object across branches, so a single canonical entry
    # covers all of them; for shared get_attrs the canonical is the branch-0
    # proxy. A non-shared canonical lifted by multiple branches is assigned
    # to the first branch that lifted it.
    def _sort_by_name(vars: Iterable[Proxy]) -> list[Proxy]:
        # [FUNCTION DOC] HELPER: Returns proxies sorted by their node name.
        return sorted(vars, key=lambda var: var.node.name)

    shared = _sort_by_name(shared_canonical)
    claimed_canonical: set[Proxy] = set(shared_canonical)
    unique_per_branch: list[list[Proxy]] = []
    for outer_to_canonical in per_branch_outer_to_canonical:
        branch_unique = []
        for outer, canonical in outer_to_canonical.items():
            if canonical in claimed_canonical:
                continue
            branch_unique.append(outer)
            claimed_canonical.add(canonical)
        unique_per_branch.append(_sort_by_name(branch_unique))

    # For N branches the merged signature has 1 + N blocks: shared (no suffix)
    # then one suffixed block per branch. Within each block proxies are
    # ordered by node name for determinism.
    def fixup_branch_inps(
        graph: torch.fx.Graph,
        branch_lifted: dict[Proxy, Proxy],
        outer_to_canonical: dict[Proxy, Proxy],
    ) -> None:
        # [FUNCTION DOC] HELPER: Rewrites one branch graph in place: inserts new placeholders for
        # the shared block then each branch's unique block before the first non-placeholder node,
        # rewiring uses from the old lifted placeholders (which are erased).
        # Reverse-map canonical -> this branch's outer proxy so we can find the
        # old placeholder to replace when emitting the shared block.
        canonical_to_branch_outer = {c: o for o, c in outer_to_canonical.items()}

        def _insert_or_replace_phs(new_args: Sequence[Proxy], name_suffix: str) -> None:
            # [FUNCTION DOC] HELPER: Creates a placeholder per proxy (name plus suffix, copying
            # metadata); if this branch lifted that proxy, redirects all uses of the old
            # placeholder to the new one and erases the old one.
            for arg in new_args:
                new_ph = graph.placeholder(arg.node.name + name_suffix)
                new_ph.meta = arg.node.meta
                # If this branch lifted `arg` (directly or via a shared
                # get_attr target), wire the new placeholder to the existing
                # uses and erase the old placeholder.
                local_outer = canonical_to_branch_outer.get(arg, arg)
                if local_outer in branch_lifted:
                    old_ph = branch_lifted[local_outer].node
                    old_ph.replace_all_uses_with(new_ph)
                    # replace_all_uses_with doesn't clean users. Clean it manually so that we could erase it.
                    old_ph.users = {}
                    graph.erase_node(old_ph)

        first_not_ph_node = next(
            node for node in graph.nodes if node.op != "placeholder"
        )
        with graph.inserting_before(first_not_ph_node):
            _insert_or_replace_phs(shared, "")
            for other_name, other_unique in zip(names, unique_per_branch):
                _insert_or_replace_phs(other_unique, "_" + other_name)

    for graph, branch_lifted, outer_to_canonical in zip(
        graphs, lifted_freevars, per_branch_outer_to_canonical
    ):
        fixup_branch_inps(graph, branch_lifted, outer_to_canonical)

    return shared, unique_per_branch


# NOTE: [HigherOrderOperator subgraph input ordering]
# The input ordering of the higher order ops is determined by the order of
# the creation of the placeholder.
# Manually created inputs are created in validate_args_and_maybe_create_graph_inputs before
# speculating subgraph.
# During subgraph speculation, we may lift closured tensors and free symbols as inputs,
# their ordering is determined by the time they are lifted: earlier lifted ones precede later
# lifted ones.
#
# Suppose the placeholders are
# O1, O2, X1, O3, O4, X2, X3, O5 where Xs are lifted phs
# The following code re-order the placeholders to
# O1, O2, O3, O4, O5, X1, X2, X3
def move_lifted_freevars_phs_to_end(
    graph: torch.fx.Graph, lifted_freevars: dict[Proxy, Proxy]
) -> None:
    # [FUNCTION DOC] Reorders placeholders in place so all lifted free-variable placeholders come
    # last (keeping relative order) after the original inputs; no-op if there are no
    # placeholders, no lifted ones, or all are lifted. Raises AssertionError if the final order
    # differs from the lifted_freevars order. Lints the graph.
    lifted_ph_set = {child_p.node for child_p in lifted_freevars.values()}

    prev_phs = [n for n in graph.nodes if n.op == "placeholder"]

    # No need to reorder when graph doesn't have args or doesn't
    # have lifted freevars or all inputs are lifted freevars.
    if (
        len(prev_phs) == 0
        or len(lifted_ph_set) == 0
        or len(prev_phs) == len(lifted_ph_set)
    ):
        return

    # Step 1: find first X1
    for x1 in prev_phs:
        if x1 in lifted_ph_set:
            break

    if x1 is None:
        raise AssertionError("x1 must not be None")
    if x1.op != "placeholder":
        raise AssertionError("x1.op must be 'placeholder'")
    # Step 2: starting from the X1, skip Xs and prepend Os before X1.
    cand_x = x1.next
    while cand_x is not None and cand_x.op == "placeholder":
        if cand_x in lifted_ph_set:
            cand_x = cand_x.next
        else:
            nxt = cand_x.next
            cand_x._remove_from_list()
            x1.prepend(cand_x)
            cand_x = nxt

    # Step 3: assert that all placeholders are in the correct order as .
    # in lifted_freevars
    after_phs = [node for node in graph.nodes if node.op == "placeholder"][
        -len(lifted_freevars) :
    ]
    if len(after_phs) != len(lifted_freevars):
        raise AssertionError(
            "Number of trailing placeholders must match number of lifted_freevars"
        )
    for child_proxy, ph in zip(lifted_freevars.values(), after_phs):
        if child_proxy.node is not ph:
            raise AssertionError(
                "The order of placeholders is different from the order of lifted_freevars"
            )

    graph.lint()


def check_aliasing_and_input_mutation(
    subtracer: "SubgraphTracer",
    graph: torch.fx.Graph,
    supports_input_mutation: bool,
    supports_aliasing: bool,
    source_target: Optional["HigherOrderOperator"],
) -> None:
    # [FUNCTION DOC] Queries the subtracer for input mutation and output aliasing and graph breaks
    # (with hints) when found and the HOP does not support them. Does nothing for supported
    # features.
    name = source_target.name if source_target else "<UNKNOWN>"
    if not supports_input_mutation:
        mutation_info = subtracer.has_input_mutation()
        if mutation_info.has_mutation:
            context = f"{mutation_info.msg} in\n {graph}"
            unimplemented(
                gb_type="Encountered input mutation during higher order op tracing",
                context=context,
                explanation=f"Higher order ops do not support input mutation. Found in {name}",
                hints=[
                    "Consider using the debug context to change user code to avoid mutation.",
                    "Please open an issue.",
                ],
            )

    if not supports_aliasing:
        aliasing_info = subtracer.has_aliasing()
        if aliasing_info.has_aliasing:
            context = f"{aliasing_info.msg} in\n {graph}"
            unimplemented(
                gb_type="Encountered aliasing during higher order op tracing",
                context=context,
                explanation=f"Higher order ops do not support aliasing. Found in {name}",
                hints=[
                    "Replace `return input` with `return input.clone()` to avoid aliasing.",
                    "Consider using the debug context to change user code to avoid aliasing.",
                    "Please open an issue.",
                ],
            )


def trace_hop_function(
    f: VariableTracker,
    tx: "InstructionTranslatorBase",
    subtracer: "SubgraphTracer",
    enable_grad: bool | None,
    restore_side_effects: bool,
    args: list[VariableTracker],
    sub_kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] Calls the traced function variable with args and kwargs under an optional
    # grad-mode override. If restore_side_effects is set, snapshots side effects first, defers
    # outer-scope mutation checks, and after the call restores the snapshot while preserving
    # runahead tensor/symvar tracking from the new state. Returns the call's output variable.
    # For autograd.Function and other legacy HOPs, we do NOT couple
    # restore_side_effects with allow_side_effects_in_hop.
    # This preserves the old behavior where:
    # - restore_side_effects=False means ctx mutations persist
    # - But non-ctx side effects still cause graph breaks (under_activation_checkpoint was False)
    enable_side_effects_with_extra_outputs = False

    autograd_ctx = (
        dynamo_enable_grad(tx, enable_grad)
        if enable_grad is not None
        else contextlib.nullcontext()
    )
    side_effects_ctx = (
        dynamo_allow_side_effects_in_hop(tx)
        if enable_side_effects_with_extra_outputs
        else contextlib.nullcontext()
    )

    # For handling side effects, we can make an argument that we don't
    # have to do anything here. The side effects infra does a good job
    # of graph breaking if we mutate any nonlocal or global variable
    # while subtracing. As a result if tracing succeeds, side effects
    # data structure will only contain read-only data structures that
    # are put there for tracking purposes.
    # But on the other hand, there is an argument that if we ever write
    # a new side effect in Dynamo which does not go through the side
    # effect infra, we can end up in bad state.
    # Therefore we restore the side effects after tracing. The catch is
    # that we have to special handle tensor variables. If we have seen a
    # nonlocal variable tensor during subtracing, we want to keep a
    # track of that tensor, so that later subtracing or the root tracer
    # itself does not create a new proxy for the already observed tensor
    # variable.
    prev_side_effects = None
    if restore_side_effects:
        prev_side_effects = tx.output.side_effects.clone()

    # When restoring side effects, defer outer-scope mutation checks so
    # context managers that flip-flop a flag don't fail immediately. After
    # tracing, we validate that all deferred mutations were nullified.
    deferred_ctx = (
        tx.output.side_effects.defer_side_effect_checks()
        if restore_side_effects
        else contextlib.nullcontext()
    )

    with autograd_ctx, side_effects_ctx, deferred_ctx:
        output = f.call_function(tx, args, sub_kwargs)

    if restore_side_effects:
        new_side_effects = tx.output.side_effects.clone()
        if prev_side_effects is None:
            raise AssertionError(
                "prev_side_effects must not be None when restoring side effects"
            )
        prev_side_effects.track_runahead_tensor_and_symvar_side_effects(
            new_side_effects
        )
        tx.output.side_effects = prev_side_effects
    return output


def trace_hop_function_with_auto_output_flattening(
    f: VariableTracker,
    tx: "InstructionTranslatorBase",
    subtracer: "SubgraphTracer",
    enable_grad: bool | None,
    allow_side_effects: bool,
    args: list[VariableTracker],
    sub_kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # [FUNCTION DOC] Calls the function variable with the args under an optional grad-mode
    # override. If allow_side_effects is true, the tracer is allowed to record side effects;
    # otherwise outer-scope mutation checks are deferred. Returns the output variable.
    autograd_ctx = (
        dynamo_enable_grad(tx, enable_grad)
        if enable_grad is not None
        else contextlib.nullcontext()
    )
    side_effects_ctx = (
        dynamo_allow_side_effects_in_hop(tx)
        if allow_side_effects
        else contextlib.nullcontext()
    )

    deferred_ctx = (
        tx.output.side_effects.defer_side_effect_checks()
        if not allow_side_effects
        else contextlib.nullcontext()
    )

    with autograd_ctx, side_effects_ctx, deferred_ctx:
        output = f.call_function(tx, args, sub_kwargs)

    return output


def get_hop_args(
    tx: "InstructionTranslatorBase",
    f: VariableTracker,
    subtracer: "SubgraphTracer",
    sub_args: list[VariableTracker],
    sub_kwargs: dict[str, VariableTracker],
    set_subgraph_inputs: SetSubgraphInputs,
    description: str,
) -> list[VariableTracker]:
    # [FUNCTION DOC] HELPER: Derives positional parameter names of f (dropped if fewer than
    # sub_args), prepares positional arguments via validate_args_and_maybe_create_graph_inputs
    # with the requested input mode, and registers kwargs values as automatic inputs. Returns
    # the prepared positional args.
    sub_args_names = maybe_positional_arg_names(f)
    # User mismatch in the number of args. Will eventually lead to an error.
    if sub_args_names is not None and len(sub_args_names) < len(sub_args):
        sub_args_names = None
    args = validate_args_and_maybe_create_graph_inputs(
        sub_args,
        subtracer,
        tx,
        set_subgraph_inputs,
        description,
        sub_args_names,
    )

    validate_args_and_maybe_create_graph_inputs(
        sub_kwargs.values(),  # type: ignore[arg-type]
        subtracer,
        tx,
        set_subgraph_inputs="automatic",
        description=description,
    )
    return args


# TODO - The eventual goal is to replace
# speculate_subgraph_with_auto_output_flattening with speculate_subgraph or
# merge them two into one. We are following a staged approach because of
# existing implementation complexity for control flow ops.
def speculate_subgraph_with_auto_output_flattening(
    tx: "InstructionTranslatorBase",
    f: VariableTracker,
    sub_args: list[VariableTracker],
    sub_kwargs: dict[str, VariableTracker] | None,
    description: str,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target: HigherOrderOperator | None = None,
    enable_grad: bool | None = None,
    # automatic: relies on Dynamo to find the used tensors and lift them as
    # inputs.
    #
    # automatic_with_forced_inputs: relies on the function arg names to create
    # a new proxy. Also, it will always INSERT a tensor placeholder as input,
    # even though it might not be used in the graph and they will also be in the
    # same order as the original function (as opposed to automatic which will
    # not insert the unused placeholder and can insert other placeholders in the
    # order they are see while tracing). This is useful for autograd.Function
    # backward where we do need to account for all the inputs of the backwards
    # to be lifted as inputs for making the fwd-bwd graph consistent.
    set_subgraph_inputs: SetSubgraphInputs = "automatic",
    # If True, exposes intermediates to subgraph outputs to allow later tensor ops to
    # access intermediates from the subgraph, this is useful for mutation
    allow_side_effects: bool = False,
    # Controls whether to filter aliased intermediates when collecting extra outputs.
    # This is only relevant when allow_side_effects=True.
    # - True: Filter out intermediates that alias with inputs or outputs (strict, for invoke_subgraph)
    # - False: Allow aliased intermediates (for checkpoint/autograd.Function which get desugared/inlined)
    #
    # Example where filtering is needed:
    #
    #   @invoke_subgraph
    #   def gn(x):
    #       view = x.view(2, 4)  # intermediate that aliases input x
    #       y = torch.sin(view)
    #       return torch.cos(view)
    #
    #   def fn(x):
    #       res = gn(x)
    #       return res + 4
    #
    # In this case, if we don't filter `view`, we would later error because some HOPs
    # have strict aliasing checks on inputs/outputs.
    #
    # This does however introduce a subtle issue when we do something like:
    #
    #   captured = []
    #
    #   @invoke_subgraph
    #   def gn(x):
    #       view = x.view(2, 4)  # intermediate that aliases input x
    #       y = torch.sin(view)
    #       captured.append(view)
    #       return torch.cos(view)
    #
    #   def fn(x):
    #       res = gn(x)
    #       return res + captured[0]
    #
    # In this case, we will not replay the side effect on `captured` in the graph,
    # which fails with a not-so-nice error. We will address this in a follow-up PR
    # because this case is rare. This is not a regression because side effects were
    # never supported for invoke_subgraph anyway.
    filter_aliased_intermediates: bool = False,
    # TODO - supports input_mutation and aliasing should be False by default for strictness
    supports_input_mutation: bool = True,
    supports_aliasing: bool = True,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer: Optional["SubgraphTracer"] = None,
) -> tuple[
    VariableTracker,  # output: The VT that Dynamo continues tracing with
    torch.fx.Graph,  # graph: The FX graph representing the subgraph computation
    dict[
        torch.fx.Proxy, torch.fx.Proxy
    ],  # lifted_freevars: Free variables lifted as inputs
    VariableTracker
    | tuple[
        VariableTracker, ...
    ],  # graph_output_vts: Tensor/symint VTs that are actual FX graph outputs
    SubgraphTracingInfo,  # tracing_info: Properties observed during subgraph tracing
]:
    # [FUNCTION DOC] Traces f into a fresh subgraph (via a subtracer) and returns (output variable,
    # FX graph, lifted free variables, tuple of graph output variables, tracing info). Only
    # tensor/symint/custom-class variables found in the output are made graph outputs; with side
    # effects allowed, tracked intermediates are appended as extra outputs. Lifts free variables,
    # moves their placeholders last, adds the output node, and checks mutation/aliasing support.
    # Raises AssertionError on invalid set_subgraph_inputs; graph breaks if kwargs are combined
    # with non-automatic inputs; re-raises Unsupported after logging an informational message.
    # Side effects: temporarily resets and restores the externally-visible-side-effects flag.
    """
    Speculate subgraph for Higher-Order Operators (HOPs) with automatic output flattening.

    ## Automatic output flattening

    For many HOPs, the representation exists only as a container for the
    subgraph. In later compiler stages or at runtime, the HOP is desugared and
    simply executes the subgraph directly, as if it were inlined. For such hops,
    we follow automatic output flattening.
    For example:
    - invoke_subgraph
    - activation checkpointing (torch.utils.checkpoint.checkpoint)
    - autograd.Function
    - nested_compile_region

    This is in contrast to control flow HOPs which do not follow this desugaring:
    - torch.cond (conditional execution based on predicate)
    - torch.while_loop (iterative execution)
    - torch.map (parallel execution over batch dimension)

    For control flow HOPs, the HOP behavior is fundamentally different from just
    running the body function once.

    ## Key Advantage: Disentangling VTs from Graph Outputs

    Desugaring simplify HOP processing by allowing us to disentangle the output
    variable trackers (VTs) from the HOP subgraph outputs. This mirrors typical
    Dynamo processing where:
    - VTs "run ahead" representing the program state for continued tracing
    - The graph is a side data structure tracking computation seen so far

    This separation is crucial for HOPs with non-proxyable outputs (e.g., custom
    user-defined objects containing tensors). The function may return complex Python
    objects for Dynamo to continue tracing, but only the tensor/symint VTs need to
    be registered as actual FX graph outputs.

    Example:
        class Foo:
            def __init__(self, a, b):
                self.a = a  # tensor
                self.b = b  # tensor

        def gn(x):
            return Foo(torch.sin(x), torch.cos(x))

        result = some_hop(gn, x)  # Returns Foo instance
        out = result.a + result.b  # Dynamo can continue tracing

    Here, `output` VT is a UserDefinedObjectVariable wrapping Foo, but
    `graph_output_vts` contains only the tensor VTs (a and b) that should be
    actual FX graph outputs. This allows Dynamo to continue tracing with the
    Foo object while the graph only needs to output the constituent tensors.

    ## Return Values

    Unlike `speculate_subgraph`, this function returns:
    - output: The VT that Dynamo continues tracing with (may be complex Python objects)
    - graph: The FX graph representing the subgraph computation
    - lifted_freevars: Free variables lifted as inputs to the subgraph
    - graph_output_vts: Only the tensor/symint VTs that are actual FX graph outputs

    The key difference is `graph_output_vts` instead of `treespec`, which gives more
    flexibility for handling non-proxyable outputs.
    """
    if sub_kwargs is None:
        sub_kwargs = {}

    if set_subgraph_inputs not in get_args(SetSubgraphInputs):
        raise AssertionError(
            "Please use one of the supported set_subgraph_inputs options."
        )

    # See NOTE [Temporary argument `set_subgraph_inputs`]
    if sub_kwargs and set_subgraph_inputs != "automatic":
        unimplemented(
            gb_type="invalid set_subgraph_inputs and sub_kwargs settings",
            context=f"set_subgraph_inputs: {set_subgraph_inputs}, sub_kwargs: {sub_kwargs}",
            explanation="`sub_kwargs` cannot be used when `set_subgraph_inputs` is not set to 'automatic'.",
            hints=[
                "Use `set_subgraph_inputs='automatic'` when passing `sub_kwargs`.",
                *graph_break_hints.USER_ERROR,
            ],
        )

    try:
        with tx.output.subtracer(source_target, tracer, description) as subtracer:
            args = get_hop_args(
                tx,
                f,
                subtracer,
                list(sub_args),
                sub_kwargs,
                set_subgraph_inputs,
                description,
            )

            # Special case - if users uses
            # `traced_with_externally_visible_side_effects`, we still need to
            # return the intermediates as outputs. However, this API gets
            # triggered during the hop tracing,  and we don't know at this point
            # of time, if the API will take into effect. To handle this, we have
            # a flag traced_with_externally_visible_side_effects (default=False)
            # that is set to True anytime
            # `traced_with_externally_visible_side_effects` is set. We reset it
            # with the old value after the hop is traced out.
            old_value = (
                tx.output.current_tracer.traced_with_externally_visible_side_effects
            )

            output = trace_hop_function_with_auto_output_flattening(
                f,
                tx,
                subtracer,
                enable_grad,
                allow_side_effects,
                args,
                sub_kwargs,
            )

            # NOTE: [Separation of graph outputs and output VTs]
            # In Dynamo (outside of speculate_subgraph), VTs and the graph are
            # separate concepts:
            # - VTs (VariableTrackers) can "run ahead" and continue Dynamo tracing
            # - The graph is just a side data structure tracking computation seen so far
            #
            # This separation is crucial for HOPs with non-proxyable outputs (e.g.,
            # custom user-defined objects containing tensors). The function may return
            # complex Python objects for Dynamo to continue tracing, but only the
            # tensor/symint VTs need to be registered as actual graph outputs.
            #
            # Example:
            #   class Foo:
            #       def __init__(self, a, b):
            #           self.a = a  # tensor
            #           self.b = b  # tensor
            #
            #   def gn(x):
            #       return Foo(torch.sin(x), torch.cos(x))
            #
            # Here, `output` VT is a UserDefinedObjectVariable wrapping Foo, but
            # `graph_output_vts` contains only the tensor VTs (a and b) that should
            # be actual FX graph outputs.
            # Collect only tensor and symint VTs that should be graph outputs.
            # We walk the output structure and extract proxyable VTs.
            graph_output_vts: list[VariableTracker] = []

            def visit(vt: VariableTracker) -> None:
                # [FUNCTION DOC] HELPER: Visitor that appends tensor, symbolic-number and custom
                # class object variables to graph_output_vts.
                if vt.is_tensor() or isinstance(
                    vt, (SymNodeVariable, CustomClassObjectVariable)
                ):
                    graph_output_vts.append(vt)

            VariableTracker.visit(visit, output, side_effects=tx.output.side_effects)

            # NOTE - [Return subgraph intermediates as subgraph outputs]
            # This helps HOPs which allow side effects. Consider the
            # following example
            #
            # def gn(x, z):
            #     o = torch.matmul(x, x) @ x
            #     out = x.sin()
            #     z.append(out)
            #     return torch.cos(torch.sin(o))

            # def fn(x):
            #     z = []
            #     out1 = torch.utils.checkpoint.checkpoint(
            #         gn,
            #         x,
            #         z,
            #         use_reentrant=False,
            #     )
            #     return out1, z[0]
            #
            # In this example, list `z` is in outer scope and gets appended
            # in the subgraph with `out`. But `out` is not an output of the
            # subgraph. This can cause issue because later on when the outer
            # graph returns `z[0]` it needs to have access to the graph node
            # `out`. To solve this problem, we just return all intermediates
            # from the subgraph.

            # TODO - Today this is supported only for AC. AC HOP gets
            # desugared in AOTDispatcher so even though subgraph has extra
            # unused outputs in Dynamo, its ok even if we don't DCE them in
            # Dynamo. As AOTDispatcher desugars/inlines the subgraph, the
            # subgraph boundary disappears. And even for AC, today this only
            # works when the skip_fwd_side_effects_in_bwd_under_checkpoint
            # flag is True, i.e., only when we allow side-effects. But, we
            # want this to be supported for other Hops as well, specifically
            # nested_compile_region and autograd.Function. Today, its safe
            # because we error out on seeing a side-effect.

            traced_externally = (
                tx.output.current_tracer.traced_with_externally_visible_side_effects
            )
            has_side_effects = (
                subtracer.side_effect_stack is not None or traced_externally
            )
            if (allow_side_effects or traced_externally) and has_side_effects:
                extra_outputs = collect_intermediate_outputs(
                    tx, subtracer, graph_output_vts, filter_aliased_intermediates
                )
                graph_output_vts = graph_output_vts + extra_outputs

            tx.output.current_tracer.traced_with_externally_visible_side_effects = (
                old_value
            )

            validate_subgraph_output_types(graph_output_vts)

            # The output proxies might not belong to this SubgraphTracer
            # (if they are free variables that were never lifted)
            # so lift them here.
            # output_proxies = output.as_proxy()
            if isinstance(graph_output_vts, list):
                output_proxies = [a.as_proxy() for a in graph_output_vts]  # type: ignore[attr-defined]
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )
                output_proxies = tuple(output_proxies)
            else:
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )

            tx.output.create_node(
                "output",
                "output",
                (subtracer.create_arg((output_proxies,))),
                {},
            )
            graph = tx.output.graph
            graph.lint()
            lifted_freevars = subtracer.lifted_freevars

            if len(lifted_freevars) > 0:
                move_lifted_freevars_phs_to_end(graph, lifted_freevars)

            check_aliasing_and_input_mutation(
                subtracer,
                graph,
                supports_input_mutation,
                supports_aliasing,
                source_target,
            )
            # Return both the output VT and the graph output VTs separately:
            # - `output`: The VT that Dynamo continues tracing with (may be
            #   complex Python objects, tuples, dicts, etc.)
            # - `graph`: The FX graph representing the subgraph computation
            # - `lifted_freevars`: Free variables lifted as inputs to the subgraph
            # - `graph_output_vts`: Only the tensor/symint VTs that are actual
            #   FX graph outputs (basically the vts associated with graph outputs)
            # - `tracing_info`: Properties observed during subgraph tracing
            tracing_info = SubgraphTracingInfo(
                side_effect_stack=subtracer.side_effect_stack,
                traced_sources=subtracer.traced_sources,
            )

            return (
                output,
                graph,
                lifted_freevars,
                tuple(graph_output_vts),
                tracing_info,
            )
    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        log.info(msg)
        log.info(ex)
        raise ex


# See NOTE [HigherOrderOperator tracing design] for details of the design
def speculate_subgraph(
    tx: "InstructionTranslatorBase",
    f: VariableTracker,
    sub_args: list[VariableTracker],
    sub_kwargs: dict[str, VariableTracker] | None,
    description: str,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target: HigherOrderOperator | None = None,
    always_restore: bool = False,
    enable_grad: bool | None = None,
    # NOTE [argument `set_subgraph_inputs`]
    # set_subgraph_inputs controls what how to construct subgraphs' placeholders from sub_args.
    # 1. if your HOP supports arbitrary inputs, use set_subgraph_inputs="automatic" (most recommended).
    # 2. if your HOP supports only Tensor and symnode inputs, use set_subgraph_inputs="flatten_manual" (recommended).
    # If sub_args contain Pytree structure (e.g. dict/list/tuple/set), the sub_args will be flattened first.
    # Then the flattened args are manually set as subgraph's placeholders.
    # 3. if your HOP must preserve inputs that are not tensor or symnode as placeholders e.g. AutogradFunctionContextVariable
    # use set_subgraph_inputs="manual" (not recommended). We do not recommend it in general because it has the
    # restriction that user need to manually control how to create placeholders and VariableTrackers for the args.
    set_subgraph_inputs: SetSubgraphInputs = "automatic",
    restore_side_effects: bool = True,
    should_flatten_outputs: bool = False,
    # if should_flatten_outputs is True, `remove_consts_from_outputs` remove the
    # const outputs from the subgraph output.
    remove_consts_from_outputs: bool = True,
    # TODO - supports input_mutation and aliasing should be False by default for strictness
    supports_input_mutation: bool = True,
    supports_aliasing: bool = True,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer: Optional["SubgraphTracer"] = None,
) -> tuple[tuple[VariableTracker, OutputSpec], torch.fx.Graph, dict[Proxy, Proxy]]:
    # [FUNCTION DOC] Traces f into a new subgraph and returns ((output variable, OutputSpec),
    # graph, lifted free variables). Realizes lazy args, prepares inputs per set_subgraph_inputs,
    # calls f (optionally with grad override and side-effect restoration). If should_flatten_outputs,
    # pytree-flattens the output into a tuple and (optionally) strips constants into the spec.
    # With always_restore it returns without creating an output node; otherwise validates output
    # types, lifts free variables, adds the output node, reorders placeholders, checks
    # mutation/aliasing and records mutated input indices on the graph. Graph breaks on kwargs with
    # non-automatic inputs; raises AssertionError on invalid input mode; re-raises Unsupported
    # after logging.
    if sub_kwargs is None:
        sub_kwargs = {}

    from .builder import SourcelessBuilder

    if set_subgraph_inputs not in get_args(SetSubgraphInputs):
        raise AssertionError(
            "Please use one of the supported set_subgraph_inputs options."
        )

    # See NOTE [Temporary argument `set_subgraph_inputs`]
    if sub_kwargs and set_subgraph_inputs != "automatic":
        unimplemented(
            gb_type="invalid set_subgraph_inputs and sub_kwargs settings",
            context=f"set_subgraph_inputs: {set_subgraph_inputs}, sub_kwargs: {sub_kwargs}",
            explanation="`sub_kwargs` cannot be used when `set_subgraph_inputs` is not set to 'automatic'.",
            hints=[
                "Use `set_subgraph_inputs='automatic'` when passing `sub_kwargs`.",
                *graph_break_hints.USER_ERROR,
            ],
        )

    try:
        # ensure guards on args get installed in parent subgraph
        f, sub_args, sub_kwargs = LazyVariableTracker.realize_all(
            (f, sub_args, sub_kwargs),
        )

        with tx.output.subtracer(source_target, tracer, description) as subtracer:
            args = get_hop_args(
                tx, f, subtracer, sub_args, sub_kwargs, set_subgraph_inputs, description
            )

            output = trace_hop_function(
                f,
                tx,
                subtracer,
                enable_grad,
                restore_side_effects,
                args,
                sub_kwargs,
            )

            treespec = None
            masks_to_filter_const_values = None
            const_values = None
            if should_flatten_outputs:
                from torch._dynamo.external_utils import filter_out_const_values

                # Flatten the speculated subgraph output.
                output, treespec = unpack_iterable(
                    tx, _make_inlined(tx, pytree.tree_flatten)(output)
                )

                # Actually, transform the list (returned by flatten) into a tuple
                # for dynamo consistency.
                output = SourcelessBuilder.create(tx, tuple).call_function(
                    tx, [output], {}
                )

                if remove_consts_from_outputs:
                    # Filter out the constants and save them into a spec. Filtering
                    # out constants makes the graph simpler for the backends. We
                    # need to ensure that after unflattening the constants are
                    # inserted back at the right positions for the Dynamo tracing to
                    # continue. This is done by filter_const_spec
                    output_proxies = output.as_proxy()
                    masks_to_filter_const_values = pytree.tree_map(
                        lambda x: not isinstance(x, torch.fx.Proxy), output_proxies
                    )
                    const_values = pytree.tree_map(
                        lambda x: None if isinstance(x, torch.fx.Proxy) else x,
                        output_proxies,
                    )
                    output = _make_inlined(tx, filter_out_const_values)(
                        output, masks_to_filter_const_values
                    )

            # TODO - clean up num_intermediate_nodes_as_outputs - we do not need
            # after AC moved to auto_output_flattening
            num_intermediate_nodes_as_outputs = 0
            # Register output to graph
            # Modeled off of compile_and_call_fx_graph
            # TODO: support pytree output
            # We check always_restore because we don't use the output or side effects of always_restore code,
            # like bwd.
            if always_restore:
                # Nothing left to do here
                return (
                    (
                        output,
                        OutputSpec(
                            treespec,  # type: ignore[arg-type]
                            masks_to_filter_const_values,
                            const_values,
                            num_intermediate_nodes_as_outputs,
                        ),
                    ),
                    tx.output.graph,
                    subtracer.lifted_freevars,
                )
            else:
                validate_subgraph_output_types(output)

                # The output proxies might not belong to this SubgraphTracer
                # (if they are free variables that were never lifted)
                # so lift them here.
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )

                tx.output.create_node(
                    "output",
                    "output",
                    (subtracer.create_arg((output_proxies,))),
                    {},
                )
                graph = tx.output.graph
                graph.lint()
                lifted_freevars = subtracer.lifted_freevars

                if len(lifted_freevars) > 0:
                    move_lifted_freevars_phs_to_end(graph, lifted_freevars)
                check_aliasing_and_input_mutation(
                    subtracer,
                    graph,
                    supports_input_mutation,
                    supports_aliasing,
                    source_target,
                )
                mutation_info = subtracer.has_input_mutation()
                graph._dynamo_mutated_input_indices = (  # pyrefly: ignore[missing-attribute]
                    mutation_info.mutated_input_indices
                )

                return (
                    (
                        output,
                        OutputSpec(
                            treespec,  # type: ignore[arg-type]
                            masks_to_filter_const_values,
                            const_values,
                            num_intermediate_nodes_as_outputs,
                        ),
                    ),
                    graph,
                    lifted_freevars,
                )

    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        log.info(msg)
        log.info(ex)
        raise ex


def make_attr(tx: "InstructionTranslatorBase", name: str) -> Proxy:
    # [FUNCTION DOC] Emits a get_attr node for the named installed attribute (e.g. a subgraph) in
    # the current graph and returns its proxy.
    node = tx.output.create_proxy(
        "get_attr",
        name,
        (),
        {},
    )
    return node


def add_hop_context(cls: type[HOP_VT_Alias]) -> type[HOP_VT_Alias]:
    # [FUNCTION DOC] Class decorator that wraps the class's call_function so failures carry HOP
    # context: UncapturedHigherOrderOpError and (when fallback to eager is allowed) Unsupported /
    # ObservedException get tagged with the HOP name once (deepest wins); when fallback is not
    # allowed, those are converted into UncapturedHigherOrderOpError. Fake tensor observed
    # exceptions pass through. Raises TypeError if _HOP_NAME or _ALLOW_FALLBACK_TO_EAGER is
    # none. Idempotent via a wrapped marker (checked on call_method). Mutates and returns cls.
    # KNOWN ISSUE: the idempotency guard checks the _hop_wrapped marker on call_method, but the
    # marker is set on the wrapped call_function, so the guard never triggers.
    """
    Class decorator that adds HOP context to exceptions raised in call_function.

    Requires the class to have _HOP_NAME and _ALLOW_FALLBACK_TO_EAGER set.
    """

    if hasattr(cls.call_method, "_hop_wrapped"):
        return cls

    if cls._HOP_NAME is None:
        raise TypeError(f"{cls.__name__} must define _HOP_NAME class attribute.")
    if cls._ALLOW_FALLBACK_TO_EAGER is None:
        raise TypeError(
            f"{cls.__name__} must define _ALLOW_FALLBACK_TO_EAGER class attribute."
        )

    original_call_function = cls.call_function

    @functools.wraps(original_call_function)
    def wrapped_call_function(self, *args: Any, **kwargs: Any) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Runs the original call_function and re-raises errors tagged with
        # the HOP name, or as UncapturedHigherOrderOpError when eager fallback is disallowed.
        try:
            return original_call_function(self, *args, **kwargs)
        except UncapturedHigherOrderOpError as e:
            if not hasattr(e, "_hop_name"):
                e._hop_name = self._HOP_NAME  # pyrefly: ignore[missing-attribute]
            raise
        except (Unsupported, ObservedException) as e:
            if isinstance(e, FakeTensorObservedException):
                raise
            # Only tag if not already tagged (reports deepest HOP only)
            if hasattr(e, "_hop_name"):
                raise

            if self._ALLOW_FALLBACK_TO_EAGER:
                # Tag the exception with HOP name for later formatting in exc.py
                # NOTE: because nested graph breaks are NOT supported on HOPs, we will
                # NEVER log a HOP graph break before running this
                e._hop_name = self._HOP_NAME  # pyrefly: ignore[missing-attribute]
                raise
            else:
                real_stack = getattr(e, "real_stack", None)
                full_msg = (
                    "This higher order operator doesn't work unless it is "
                    "captured completely with torch.compile. Got graph break/error:"
                    f"\n\n{str(e)}"
                )
                exc = UncapturedHigherOrderOpError(full_msg, real_stack)
                exc._hop_name = self._HOP_NAME  # pyrefly: ignore[missing-attribute]
                raise exc.with_traceback(e.__traceback__) from None

    wrapped_call_function._hop_wrapped = True  # pyrefly: ignore[missing-attribute]
    cls.call_function = wrapped_call_function
    return cls


class TorchHigherOrderOperatorVariable(VariableTracker):
    # [CLASS DOC] Base variable tracker wrapping a HigherOrderOperator in self.value. Subclasses
    # set _HOP_NAME (for error context) and _ALLOW_FALLBACK_TO_EAGER, are automatically decorated
    # by add_hop_context on definition, and implement _call_function. The make factory picks
    # the subclass from a registry by operator name (BaseHOP instances use BaseHOPVariable).
    # Subclasses should set _HOP_NAME to enable automatic HOP context in error messages
    _HOP_NAME: str | None = None
    # Set to False for HOPs that hard error on graph break (e.g., cond, map, scan); otherwise
    # HOPs will fall back to eager.
    _ALLOW_FALLBACK_TO_EAGER: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # [FUNCTION DOC] Applies add_hop_context to each new subclass at class creation.
        super().__init_subclass__(**kwargs)
        add_hop_context(cls)

    def __init__(
        self, value: HigherOrderOperator, source: Source | None = None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Stores the wrapped HigherOrderOperator and source.
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    @staticmethod
    def make(
        value: HigherOrderOperator, source: Source | None = None, **kwargs: Any
    ) -> "TorchHigherOrderOperatorVariable":
        # [FUNCTION DOC] Factory: looks up the variable class by the operator's name in the
        # registry and instantiates it; falls back to BaseHOPVariable for BaseHOP instances;
        # otherwise graph breaks as unsupported.
        variable_class = _hop_name_to_variable_class.get(value.__name__)
        if variable_class is not None:
            return variable_class(value, source, **kwargs)

        from torch._higher_order_ops import BaseHOP

        if isinstance(value, BaseHOP):
            return BaseHOPVariable(value, source, **kwargs)
        unimplemented(
            gb_type="unsupported HigherOrderOperator",
            context=str(value),
            explanation=f"Unable to create higher order operator variable for {value.__name__}.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich comparison treating this variable as a python constant (the wrapped
        # operator); delegates to python_constant_richcompare_impl.
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Entry point for calling the HOP during tracing: dispatches to a
        # __torch_function__ override if any argument has one, otherwise calls the subclass's
        # _call_function. Returns the resulting variable.
        from .torch_function import can_dispatch_torch_function, dispatch_torch_function

        if can_dispatch_torch_function(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)

        return self._call_function(tx, args, kwargs)

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Default implementation that graph breaks; subclasses override to trace.
        unimplemented(
            gb_type="unsupported HigherOrderOperator function call",
            context=str(self.value),
            explanation=f"Unable to trace calling higher order operator variable for {self.value.__name__}.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )

    def as_python_constant(self) -> HigherOrderOperator:
        # [FUNCTION DOC] Returns the wrapped HigherOrderOperator.
        return self.value


class CustomFunctionHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the custom_function_call HOP; forwards calls to the operator's own
    # call method as a user method so Dynamo traces it directly.
    """
    Wraps torch._functorch.autograd_function.custom_function_call
    """

    _HOP_NAME = "torch.ops.higher_order.custom_function_call"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Builds a user method variable for the operator's call method bound to a
        # user-defined object wrapper (sourced from this variable's source) and calls it with
        # args/kwargs. Raises AssertionError if there is no source.
        if self.source is None:
            raise AssertionError("source must not be None")
        return torch._dynamo.variables.UserMethodVariable(
            self.value.__call__.__func__,
            torch._dynamo.variables.UserDefinedObjectVariable(
                self.value, source=self.source
            ),
            source=AttrSource(self.source, "__call__"),
        ).call_function(tx, args, kwargs)


class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for torch.cond. Hard-errors (no eager fallback) on graph breaks. Input
    # mutation and aliasing support flags are recomputed per call from the grad mode.
    _HOP_NAME = "torch.cond"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces cond(pred, true_fn, false_fn, operands). Moves keyword args into
        # positional ones, graph breaks on bad arity/predicate/operand types. A python constant
        # predicate (with a warning) just inlines the chosen branch. Otherwise speculates both
        # branches (flattened outputs, tensors/int constants only, same output tree structure
        # required), merges lifted free variables, installs both subgraphs and emits the cond call
        # with shared plus per-branch unique extra operands. Returns the unflattened output.
        from . import ListVariable

        self.supports_input_mutation = not torch.is_grad_enabled()
        self.supports_aliasing = not torch.is_grad_enabled()

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        for i, k in enumerate(["pred", "true_fn", "false_fn", "operands"]):
            if v := kwargs.pop(k, None):
                if i != len(args):
                    raise AssertionError(
                        "did not provide the right number of non-keyword args"
                    )
                args.append(v)

        # TODO(voz): Support fake tensor dispatch for recursive
        # ops - see torch/dispatch/_dispatcher.py
        if len(args) != 4 or kwargs:
            unimplemented(
                gb_type="torch.cond: improper args/kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"torch.cond expects 4 positional arguments (got {len(args)}) "
                f"and no keyword arguments (got {len(kwargs)}) "
                "Usage: cond(pred, cond_fn, body_fn, operands)",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # Specialize into one of the branches since pred is constant
        pred, true_fn, false_fn, operands = args
        if type(args[0]) is ConstantVariable:
            warnings.warn(
                "Pred is a Python constant. When used with torch.cond, it specializes on one of the branches."
                " If you want torch.cond to preserve two branches, please make the predicate a boolean tensor or a SymBool.",
                UserWarning,
            )
            if pred.as_python_constant():
                return true_fn.call_function(tx, unpack_iterable(tx, operands), {})
            else:
                return false_fn.call_function(tx, unpack_iterable(tx, operands), {})

        # predicate
        if type(pred.realize()) not in (
            ConstantVariable,
            TensorVariable,
            SymNodeVariable,
        ):
            unimplemented(
                gb_type="torch.cond: improper predicate",
                context=str(pred),
                explanation="Expected `pred` to be a bool or a boolean tensor with a single item "
                f"but got {str(type(pred))} with original python type {str(pred.python_type())}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # operands
        if not isinstance(operands, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.cond: improper operands",
                context=str(operands),
                explanation="Expected `operands` to be a list/tuple "
                f"but got {operands.python_type()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        operands_seq = unpack_iterable(tx, operands)
        if not only_consist_of(
            operands, (TensorVariable, ConstantVariable, SymNodeVariable)
        ):
            unimplemented(
                gb_type="torch.cond: improper operands contents",
                context=str(operands),
                explanation="Expected `operands` to be a list/tuple of pytrees that only consists of tensor leaves.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # branches
        _check_supported_callable_arg(tx, true_fn, "true_fn")
        _check_supported_callable_arg(tx, false_fn, "false_fn")

        # Our strategy for tracing the true/false branches of cond
        # are to checkpoint our graphstate, run the true branch,
        # roll it back to the checkpoint, and run the false
        # branch, and then merge the graphstates.  Well, perhaps
        # "merge" is too strong a word: we mostly assert that
        # the resulting graphstates have to be the same.
        #
        # We only permit guards to diverge (we union the guards from
        # both branches).  In particular, this means that side
        # effects are NOT permitted inside true/false branches; this
        # would be difficult to implement, because of the path
        # explosion problem.

        def speculate_branch(
            branch: bool,
        ) -> tuple[VariableTracker, OutputSpec, torch.fx.Graph, dict[Proxy, Proxy]]:
            # [FUNCTION DOC] HELPER: Speculates the true (flag true) or false branch with the
            # operands, bumps the fake mode epoch, and graph breaks unless outputs are tensors or
            # int constants. Returns (output, spec, graph, lifted free variables).
            # NB: 0 is predicate
            ix = 1 if branch else 2
            if self._HOP_NAME is None:
                raise AssertionError("_HOP_NAME must be set")
            # TODO: Support kwargs
            (
                (ret_val, ret_spec),
                ret_graph,
                ret_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                args[ix],
                operands_seq,
                {},
                self._HOP_NAME,
                source_target=self.value,
                should_flatten_outputs=True,
                # TODO - removing consts from control flow ops need more work
                remove_consts_from_outputs=False,
                supports_input_mutation=self.supports_input_mutation,
                supports_aliasing=self.supports_aliasing,
            )

            # need to ensure we increase epoch so we don't memoize unbacked bindings
            # across different subgraphs which can interfere with runtime assertion
            # generation.
            if tx.fake_mode is None:
                raise AssertionError("tx.fake_mode must not be None")
            tx.fake_mode.epoch += 1

            if not only_consist_of(ret_val, (TensorVariable, ConstantVariable)):
                unimplemented(
                    gb_type="torch.cond: unsupported branch return type",
                    context=str(ret_val),
                    explanation="Expected branches to return a possibly nested pytree of tensors or constant ints.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            for ret in unpack_iterable(tx, ret_val):
                if ret.is_python_constant() and not isinstance(
                    ret.as_python_constant(), int
                ):
                    unimplemented(
                        gb_type="torch.cond: unsupported branch return type (constant non-int)",
                        context=str(ret_val),
                        explanation="Constants returned from branches must be ints.",
                        hints=[
                            *graph_break_hints.USER_ERROR,
                        ],
                    )
            return ret_val, ret_spec, ret_graph, ret_lifted_freevars

        (true_r, true_spec, true_graph, true_lifted_freevars) = speculate_branch(True)
        true_nn_modules = dict(tx.output.nn_modules)

        (
            false_r,
            false_spec,
            false_graph,
            false_lifted_freevars,
        ) = speculate_branch(False)
        false_nn_modules = dict(tx.output.nn_modules)

        same_spec = _make_inlined(tx, pytree.TreeSpec.__eq__)(
            true_spec.treespec, false_spec.treespec
        ).as_python_constant()
        # 3.14: NotImplemented cannot be converted to bool
        if same_spec is not NotImplemented and not same_spec:
            unimplemented(
                gb_type="torch.cond: differing branch outputs",
                context=f"true_spec: {true_spec.treespec}, false_spec: {false_spec.treespec}, same_spec: {same_spec}",
                explanation="Expected branches to return the same pytree structure.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        shared, (unique_true, unique_false) = _merge_graph_inputs(
            [true_graph, false_graph],
            [true_lifted_freevars, false_lifted_freevars],
            ["true_branch", "false_branch"],
        )

        true_name = tx.output.install_subgraph(
            "cond_true",
            torch.fx.GraphModule(true_nn_modules, true_graph),
        )
        false_name = tx.output.install_subgraph(
            "cond_false",
            torch.fx.GraphModule(false_nn_modules, false_graph),
        )

        true_node = make_attr(tx, true_name)
        false_node = make_attr(tx, false_name)

        p_args = (
            pred.as_proxy(),
            true_node,
            false_node,
            tuple(shared + unique_true + unique_false),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.cond,
            p_args,
            {},
            None,
            true_spec,
            true_r,
        )


class SwitchHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for torch.switch (N-way generalization of cond). Hard-errors on graph
    # breaks; input mutation support depends on grad mode.
    _HOP_NAME = "torch.switch"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces switch(index, branches, operands). Normalizes keyword args, graph
        # breaks on improper arity, index, operands or branches (must be a non-empty list/tuple of
        # callables). A python constant index (with a warning) selects and inlines one branch.
        # Otherwise speculates every branch, requires identical output trees, merges lifted free
        # variables across all branches, installs per-branch subgraphs and emits the switch call.
        from torch._higher_order_ops.switch import _get_branch

        from . import ListVariable

        self.supports_input_mutation = not torch.is_grad_enabled()

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        for i, k in enumerate(["index", "branches", "operands"]):
            if v := kwargs.pop(k, None):
                if i != len(args):
                    raise AssertionError(
                        "did not provide the right number of non-keyword args"
                    )
                args.append(v)

        if len(args) != 3 or kwargs:
            unimplemented(
                gb_type="torch.switch: improper args/kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"torch.switch expects 3 positional arguments (got {len(args)}) "
                f"and no keyword arguments (got {len(kwargs)}) "
                "Usage: switch(index, branches, operands)",
                hints=[*graph_break_hints.USER_ERROR],
            )

        index, branches, operands = args
        if type(index) is ConstantVariable:
            # Specialize into one of the branches since index is constant.
            warnings.warn(
                "Index is a Python constant. When used with torch.switch, it specializes on one of the branches."
                " If you want torch.switch to preserve the branches, please make the index a tensor or SymInt.",
                UserWarning,
            )
            idx = index.as_python_constant()
            branch_fns = branches.unpack_var_sequence(tx)
            return _get_branch(branch_fns, idx).call_function(
                tx, operands.unpack_var_sequence(tx), {}
            )

        # index
        if type(index.realize()) not in (
            ConstantVariable,
            TensorVariable,
            SymNodeVariable,
        ):
            unimplemented(
                gb_type="torch.switch: improper index",
                context=str(index),
                explanation="Expected index to be an int or a tensor with a single item "
                f"but got {str(type(index))} with original python type {str(index.python_type())}.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        # operands
        if not isinstance(operands, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.switch: improper operands",
                context=str(operands),
                explanation=f"Expected operands to be a list/tuple "
                f"but got {operands.python_type()}.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        operands_seq = operands.unpack_var_sequence(tx)
        if not only_consist_of(
            operands, (TensorVariable, ConstantVariable, SymNodeVariable)
        ):
            unimplemented(
                gb_type="torch.switch: improper operands contents",
                context=str(operands),
                explanation="Expected `operands` to be a list/tuple of pytrees that only consists of tensor leaves.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        # branches
        if not isinstance(branches, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.switch: improper branches",
                context=str(branches),
                explanation="Expected branches to be a list/tuple of callables.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        branch_fns = branches.unpack_var_sequence(tx)
        if len(branch_fns) == 0:
            unimplemented(
                gb_type="torch.switch: empty branches",
                context=str(branches),
                explanation="Expected at least one branch.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        for i, branch_fn in enumerate(branch_fns):
            _check_supported_callable_arg(tx, branch_fn, f"branch{i}")

        # Our strategy for tracing the branches of switch mirrors cond: each
        # branch is speculated in isolation and the resulting graphs are
        # merged, asserting the output specs agree. Side effects are not
        # permitted inside branches because of the path-explosion problem.
        def speculate_branch(
            func,
        ) -> tuple[VariableTracker, OutputSpec, torch.fx.Graph, dict[Proxy, Proxy]]:
            # [FUNCTION DOC] HELPER: Speculates one branch callable with the operands, bumps the
            # fake mode epoch, and graph breaks unless leaves are tensors or int/none constants.
            # Returns (output, spec, graph, lifted free variables).
            if self._HOP_NAME is None:
                raise AssertionError("_HOP_NAME must be set")
            (
                (ret_val, ret_spec),
                ret_graph,
                ret_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                func,
                operands_seq,
                {},
                self._HOP_NAME,
                source_target=self.value,
                should_flatten_outputs=True,
                remove_consts_from_outputs=False,
                supports_input_mutation=self.supports_input_mutation,
                supports_aliasing=self.supports_aliasing,
            )

            # need to ensure we increase epoch so we don't memoize unbacked bindings
            # across different subgraphs which can interfere with runtime assertion
            # generation.
            if tx.fake_mode is None:
                raise AssertionError("tx.fake_mode must not be None")
            tx.fake_mode.epoch += 1

            if not only_consist_of(
                ret_val, (TensorVariable, ConstantVariable), allow_none=True
            ):
                unimplemented(
                    gb_type="torch.switch: unsupported branch return type",
                    context=str(ret_val),
                    explanation="Expected branches to return a possibly nested pytree of tensor or constant int/None leaves.",
                    hints=[*graph_break_hints.USER_ERROR],
                )
            for ret in ret_val.unpack_var_sequence(tx):
                if ret.is_python_constant():
                    const = ret.as_python_constant()
                    # Python floats in branch outputs are blocked upstream by
                    # validate_subgraph_output_types (shared HOP gate); we
                    # mirror the whitelist here to keep the error specific to
                    # torch.switch. bool is implicitly accepted via the int check.
                    if not (isinstance(const, int) or const is None):
                        unimplemented(
                            gb_type="torch.switch: unsupported branch return type (constant)",
                            context=str(ret_val),
                            explanation="Constants returned from branches must be int or None.",
                            hints=[*graph_break_hints.USER_ERROR],
                        )
            return ret_val, ret_spec, ret_graph, ret_lifted_freevars

        branch_states: list[Any] = []
        branch_nn_modules: list[Any] = []
        for f in branch_fns:
            branch_states.append(speculate_branch(f))
            branch_nn_modules.append(dict(tx.output.nn_modules))
        branch_rets, branch_specs, branch_graphs, branch_lifted_freevars = zip(
            *branch_states
        )

        for i, (spec_a, spec_b) in enumerate(itertools.pairwise(branch_specs)):
            same_spec = _make_inlined(tx, pytree.TreeSpec.__eq__)(
                spec_a.treespec, spec_b.treespec
            ).as_python_constant()
            # 3.14: NotImplemented cannot be converted to bool
            if same_spec is not NotImplemented and not same_spec:
                unimplemented(
                    gb_type="torch.switch: differing branch outputs",
                    context=f"branch{i}: {spec_a.treespec}, branch{i + 1}: {spec_b.treespec}, same_spec: {same_spec}",
                    explanation="Expected branches to return the same pytree structure.",
                    hints=[*graph_break_hints.USER_ERROR],
                )

        # Merge lifted freevars across all branches so they all share the same
        # placeholder signature. Branches may capture different free variables
        # (e.g. different nn module attrs), so we union them and add missing
        # placeholders to any branch graph that lacks them.
        branch_names_for_merge = [f"branch{i}" for i in range(len(branch_fns))]
        shared, unique_per_branch = _merge_graph_inputs(
            list(branch_graphs),
            list(branch_lifted_freevars),
            branch_names_for_merge,
        )
        all_lifted = shared + [u for branch in unique_per_branch for u in branch]

        branch_names = [
            tx.output.install_subgraph(
                f"switch_branch{i}",
                GraphModule(branch_nn_modules[i], graph),
            )
            for i, graph in enumerate(branch_graphs)
        ]

        branch_nodes = [make_attr(tx, name) for name in branch_names]

        p_args = (
            index.as_proxy(),
            branch_nodes,
            tuple(all_lifted),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.switch,
            p_args,
            {},
            None,
            branch_specs[0],
            branch_rets[0],
        )


class CallTorchbindHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the call_torchbind HOP bound to a script object variable and a
    # method name; calling it emits a graph call of the HOP on the object, method name and args.
    _HOP_NAME = "torch.ops.higher_order.call_torchbind"

    def __init__(
        self,
        hop: HigherOrderOperator,
        source: Source | None,
        script_obj_var: Any,
        method_name: str,
    ) -> None:
        # [FUNCTION DOC] Stores the script object variable and the method name.
        super().__init__(hop, source)
        self.script_obj_var = script_obj_var
        self.method_name = method_name

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Realizes the args, converts them to proxies, emits a call_function node
        # of the HOP with the script object proxy, method name and args, and returns the wrapped
        # result variable (no example value supplied).
        from .builder import wrap_fx_proxy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        args_proxy = [arg.as_proxy() for arg in args]
        kwargs_proxy = {k: v.as_proxy() for k, v in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(
                    [self.script_obj_var.as_proxy(), self.method_name] + args_proxy
                ),
                kwargs=kwargs_proxy,
            ),
        )


def validate_subgraph_output_types(
    output: VariableTracker | list[VariableTracker],
) -> None:
    # [FUNCTION DOC] Checks all non-tensor leaves (none allowed) of the output structure are int or
    # bool symbolic numbers/constants or custom class objects; graph breaks otherwise.
    """Verify that the output of the subgraph is a tensor,
    int, bool, SymBool, or SymInt.
    """
    from . import TensorVariable

    if non_tensor_output := find_mismatched_vars(
        output, TensorVariable, allow_none=True
    ):
        for out in non_tensor_output:
            if (
                (isinstance(out, SymNodeVariable) and out.python_type() in (int, bool))
                or (
                    out.is_python_constant()
                    and isinstance(out.as_python_constant(), (int, bool))
                )
                or isinstance(out, CustomClassObjectVariable)
            ):
                continue
            unimplemented(
                gb_type="HOP body output unsupported",
                context=f"non-tensor outputs: {non_tensor_output}",
                explanation="HigherOrderOperator body's output must consist of tensors or ints/bools only "
                f"but got {out.python_type()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )


class WhileLoopHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for torch.while_loop (carried outputs). Hard-errors on graph breaks.
    _HOP_NAME = "torch.while_loop"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Sets input mutation support from grad mode and delegates to
        # _call_while_loop with stack_output disabled. Raises AssertionError if _HOP_NAME is unset.
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        self.supports_input_mutation = not torch.is_grad_enabled()
        return _call_while_loop(
            self, tx, args, kwargs, stack_output=False, hop_name=self._HOP_NAME
        )


class WhileLoopStackOutputHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the stack-output while_loop variant (outputs of every iteration
    # stacked). Hard-errors on graph breaks; mutation/aliasing unsupported.
    _HOP_NAME = "torch.while_loop"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Delegates to _call_while_loop with stack_output enabled. Raises
        # AssertionError if _HOP_NAME is unset.
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        return _call_while_loop(
            self, tx, args, kwargs, stack_output=True, hop_name=self._HOP_NAME
        )


class AssociativeScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for associative_scan. Hard-errors on graph breaks; mutation and
    # aliasing unsupported.
    _HOP_NAME = "torch.ops.higher_order.associative_scan"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces associative_scan(combine_fn, xs, additional_inputs). Validates
        # callable combine_fn and tensor-only list/tuple xs and additional inputs (graph breaks
        # otherwise), speculates combine_fn on first-slice copies of xs (twice) plus cloned extra
        # inputs, flattens the graph output, checks output tree equals the xs tree and tensor
        # metadata matches, verifies under fake mode that combine_fn is pointwise (raises
        # RuntimeError otherwise), installs the subgraph and emits the call. Returns the
        # unflattened result using the xs tree spec.
        from torch._higher_order_ops.utils import first_slice_copy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        def arg_extractor(
            combine_fn: VariableTracker,
            xs: VariableTracker,
            additional_inputs: VariableTracker,
        ) -> tuple[VariableTracker, VariableTracker, VariableTracker]:
            # [FUNCTION DOC] HELPER: Binds the call's args/kwargs to named parameters and returns
            # them as a tuple; a wrong arity raises a Python call error.
            return combine_fn, xs, additional_inputs

        combine_fn, xs, additional_inputs = arg_extractor(*args, **kwargs)

        if args[0].python_type() is functools.partial:
            # This is the standard case when the user calls the frontend
            # and the frontend invokes dynamo
            if len(args) != 2:
                unimplemented(
                    gb_type="torch.associative_scan: improper args",
                    context=f"args: {args}",
                    explanation=f"torch.associative_scan expects 2 positional arguments (got {len(args)}) "
                    "Usage: associative_scan(combine_fn, xs)",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

            xs_treespec = args[0].keywords["spec"]

            # combine_fn input check
            # We need to get the pure combine_fn from the functools.partial
            _check_supported_callable_arg(
                tx,
                combine_fn.keywords["combine_fn"],  # type: ignore[attr-defined]
                "combine_fn",
            )
        else:
            # This case is hit during re-tracing, for example in export tests
            # In this case, the combine_fn is a callable and not a functools.partial
            xs_treespec = _make_inlined(tx, pytree.tree_structure)(xs)

            _check_supported_callable_arg(tx, combine_fn, "combine_fn")

        # xs input check
        if not isinstance(xs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.associative_scan: improper xs",
                context=str(xs),
                explanation=f"Expected xs to be a list/tuple but got {xs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        xs_vars = unpack_iterable(tx, xs)
        _check_all_tensorvariable(xs_vars)

        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.associative_scan: improper additional_inputs",
                context=str(additional_inputs),
                explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        additional_inputs_vars = unpack_iterable(tx, additional_inputs)
        _check_all_tensorvariable(additional_inputs_vars)

        # Trace the subgraph
        # The sub_args is a slice of original input, e.g. if input.size is (3, 4), and scan dim=0
        # the sub_args shape will be (4, ).
        with discard_graph_changes(tx):
            sub_args = [
                _make_inlined(tx, first_slice_copy)(leaf)
                for leaf in itertools.chain(xs_vars, xs_vars)
            ]
            sub_args_additional_inputs = [
                t.call_method(tx, "clone", args=[], kwargs={})
                for t in additional_inputs_vars
            ]

        sub_args = sub_args + sub_args_additional_inputs
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        (
            (combine_result, _combine_spec),
            combine_graph,
            combine_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            combine_fn,
            sub_args,
            sub_kwargs={},
            description=self._HOP_NAME,
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        # Ensure that the output of scan is a flattened list of elements,
        # because downstream operations assume that the output of HOPs
        # is flattened
        output_node = combine_graph.find_nodes(op="output")[0]
        output_node.args = (pytree.tree_leaves(output_node.args),)
        combine_graph.lint()

        # Collect the results from the combine_fn
        results, _combine_treespec = unpack_iterable(
            tx, _make_inlined(tx, pytree.tree_flatten)(combine_result)
        )

        # Check whether the combine_fn returns one child tree for the output.
        if _combine_treespec.as_python_constant().num_leaves < 1:
            unimplemented(
                gb_type="torch.associative_scan: combine_fn improper number of leaves",
                context=str(_combine_treespec.as_python_constant()),
                explanation="combine_fn needs to produce one pytree for the output "
                f"but combine_fn produces the pytree {_combine_treespec.as_python_constant()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # Check whether the outs produced by combine_fn has the same treespec as xs
        # We need to have this check this way, because in case init is a TreeSpec and carry
        # but carry is only a LeafSpec, these two cannot be compared correctly.
        if (
            xs_treespec.as_python_constant().is_leaf()
            != _combine_treespec.as_python_constant().is_leaf()
        ) or not _make_inlined(tx, pytree.TreeSpec.__eq__)(
            xs_treespec, _combine_treespec
        ).as_python_constant():
            unimplemented(
                gb_type="torch.associative_scan: mismatched input/output tree structure",
                context=f"xs: {xs_treespec.as_python_constant()}, output: {_combine_treespec.as_python_constant()}",
                explanation="The tree structure of the xs and the outs of the combine_fn are expected to be identical, but got "
                f"xs: {xs_treespec.as_python_constant()} vs output: {_combine_treespec.as_python_constant()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            [_make_inlined(tx, first_slice_copy)(t) for t in xs_vars],
            results.items,  # type: ignore[attr-defined]
            "initial_xs",
            "combine_fn_output",
            include_contiguity=False,
        )

        combine_gm = torch.fx.GraphModule(dict(tx.output.nn_modules), combine_graph)
        combine_freevars_proxy = tuple(combine_lifted_freevars.keys())

        # Compute the proxies for the input check
        proxy_vars_inputcheck = (
            tuple(sarg.as_proxy() for sarg in sub_args) + combine_freevars_proxy
        )

        from torch._higher_order_ops.utils import _maybe_fake_tracing
        from torch._inductor.utils import is_pointwise_use

        if tx.fake_mode is None:
            raise AssertionError("tx.fake_mode must not be None")
        with tx.fake_mode:
            sub_args_fake = [
                (
                    leaf.node.meta["example_value"].clone()
                    if hasattr(leaf.node.meta["example_value"], "clone")
                    else leaf.node.meta["example_value"]
                )
                for leaf in pytree.tree_leaves(proxy_vars_inputcheck)
            ]
            pre_dispatch = False

            fx = _maybe_fake_tracing(
                combine_gm, sub_args_fake, pre_dispatch=pre_dispatch
            )

            for node in fx.graph.nodes:
                # Check that the combine_fn is pointwise, if combine_mode='pointwise'
                if not all(
                    is_pointwise_use(use) or use.op == "output" for use in node.users
                ):
                    raise RuntimeError(
                        "For combine_mode='pointwise', the combine_fn needs to be pointwise"
                    )

        combine_fn_name = tx.output.install_subgraph(
            "associative_scan_combine_fn", combine_gm
        )

        # Compute the proxies
        xs_proxy = xs.as_proxy()
        combine_freevars_proxy = tuple(combine_lifted_freevars.keys())
        additional_inputs_proxy = additional_inputs.as_proxy() + combine_freevars_proxy

        p_args = (
            make_attr(tx, combine_fn_name),
            xs_proxy,
            additional_inputs_proxy,
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.associative_scan,
            p_args,
            {},
            None,
            OutputSpec(xs_treespec),  # type: ignore[arg-type]
            None,
        )


class ScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for scan. Hard-errors on graph breaks; input mutation support depends
    # on grad mode (only additional inputs may be mutated); aliasing unsupported.
    _HOP_NAME = "torch.ops.higher_order.scan"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces scan(combine_fn, init, xs, additional_inputs). combine_fn is either
        # a partial wrapper of the user function or an already normalized graph module. Validates
        # list/tuple inputs of tensors, speculates combine_fn on cloned init, first slices of xs and
        # cloned extras, flattens the graph output, derives carry and per-step outputs, checks carry
        # metadata equals init, graph breaks if combine_fn mutates init or xs, maps mutated extra
        # inputs to parent indices (passed as mutated_arg_indices), installs the subgraph and emits
        # the scan call. Returns the unflattened output.
        from torch._higher_order_ops.scan import _extract_carry_and_out
        from torch._higher_order_ops.utils import first_slice_copy

        self.supports_input_mutation = not torch.is_grad_enabled()

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        # combine_fn input check
        def _check_combine_fn_is_normalized(combine_fn_var: VariableTracker) -> bool:
            # [FUNCTION DOC] HELPER: Graph breaks unless the variable is a module variable or a
            # functools partial; returns true for the module (normalized graph module) case.
            if not isinstance(
                combine_fn_var,
                (
                    variables.nn_module.NNModuleVariable,
                    variables.nn_module.UnspecializedNNModuleVariable,
                    variables.FunctoolsPartialVariable,
                ),
            ):
                unimplemented(
                    gb_type="torch.scan: improper combine_fn",
                    context=str(combine_fn_var),
                    explanation="Expected combine_fn to be wrapped as functools.partial in scan user-facing api "
                    f"or a graph module if we're re-exporting but got {combine_fn_var.python_type()}.",
                    hints=[
                        *graph_break_hints.DIFFICULT,
                    ],
                )
            return isinstance(
                combine_fn_var,
                (
                    variables.nn_module.NNModuleVariable,
                    variables.nn_module.UnspecializedNNModuleVariable,
                ),
            )

        def arg_extractor(
            combine_fn: VariableTracker,
            init: VariableTracker,
            xs: VariableTracker,
            additional_inputs: VariableTracker,
        ) -> tuple[VariableTracker, VariableTracker, VariableTracker, VariableTracker]:
            # [FUNCTION DOC] HELPER: Binds the call's args/kwargs to named parameters and returns
            # them as a tuple.
            return combine_fn, init, xs, additional_inputs

        combine_fn, init, xs, additional_inputs = arg_extractor(*args, **kwargs)
        init_vars = unpack_iterable(tx, init)
        xs_vars = unpack_iterable(tx, xs)
        additional_inputs_vars = unpack_iterable(tx, additional_inputs)

        # combine_fn input check
        combine_fn_is_normalized = _check_combine_fn_is_normalized(combine_fn)
        if combine_fn_is_normalized:
            combine_gm = combine_fn.value  # type: ignore[attr-defined]
            if not isinstance(combine_gm, torch.fx.GraphModule):
                raise AssertionError(
                    f"Expected combine_gm to be a GraphModule, got {combine_fn}, {combine_gm}"
                )
        else:
            # combine_fn input check
            # We need to get the pure combine_fn from the functools.partial
            _check_supported_callable_arg(
                tx,
                combine_fn.keywords["combine_fn"],  # type: ignore[attr-defined]
                "combine_fn",
            )
        # xs input check
        if not isinstance(xs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.scan: improper xs",
                context=str(xs),
                explanation=f"Expected xs to be a list/tuple but got {xs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        # init input check
        if not isinstance(init, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.scan: improper init",
                context=str(init),
                explanation=f"Expected init to be a list/tuple with at least one element but got {init.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        if len(init_vars) == 0:
            unimplemented(
                gb_type="torch.scan: no init leaves",
                context="",
                explanation="Expected init leaves.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.scan: improper additional_inputs",
                context=str(additional_inputs),
                explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        _check_all_tensorvariable(init_vars)
        _check_all_tensorvariable(xs_vars)
        _check_all_tensorvariable(additional_inputs_vars)

        with discard_graph_changes(tx):
            sub_args_init = [
                ini.call_method(tx, "clone", args=[], kwargs={}) for ini in init_vars
            ]
            # The sub_args_inp is a slice of original input, e.g. if input.size is (3, 4), and scan dim=0
            # the sub_args_inp shape will be (4, ).
            sub_args_inp = [_make_inlined(tx, first_slice_copy)(inp) for inp in xs_vars]
            sub_args_additional_inputs = [
                t.call_method(tx, "clone", args=[], kwargs={})
                for t in additional_inputs_vars
            ]

        sub_args = sub_args_init + sub_args_inp + sub_args_additional_inputs
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        (
            (combine_result, _combine_spec),
            combine_graph,
            combine_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            combine_fn,
            sub_args,
            sub_kwargs={},
            description=self._HOP_NAME,
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        combine_mutated_inputs = set(
            getattr(combine_graph, "_dynamo_mutated_input_indices", ())
        )

        # Ensure that the output of scan is a flattened list of elements,
        # because downstream operations assume that the output of HOPs
        # is flattened
        output_node = combine_graph.find_nodes(op="output")[0]
        output_node.args = (pytree.tree_leaves(output_node.args),)
        combine_graph.lint()
        combine_freevars_proxy = list(combine_lifted_freevars.keys())
        combine_result_vars = unpack_iterable(tx, combine_result)

        if combine_fn_is_normalized:
            carry_vars, out_vars = _extract_carry_and_out(
                combine_result_vars, len(init_vars)
            )
        else:
            if len(combine_result_vars) != 2:
                unimplemented(
                    gb_type="torch.scan: improper combine_fn number of returns",
                    context=str(combine_result_vars),
                    explanation=f"Expect combine_fn to return a tuple (next_carry, y) but got {combine_result_vars}.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            carry_tree, out_vars = combine_result_vars
            carry_vars, _ = unpack_iterable(
                tx, _make_inlined(tx, pytree.tree_flatten)(carry_tree)
            )
            carry_vars = unpack_iterable(tx, carry_vars)
            out_vars = unpack_iterable(
                tx, _make_inlined(tx, pytree.tree_leaves)(out_vars)
            )

            # additional output checking
            _combine_spec = OutputSpec(
                _make_inlined(tx, pytree.tree_structure)(combine_result)  # type: ignore[arg-type]
            )

            check_meta_consistency_vt(
                init_vars,
                carry_vars,
                "init",
                "carry",
            )

        # Check meta data of carries and inits. If we pass this stage, we are sure that the init and carries
        # have the same tree structure.
        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            init_vars,
            carry_vars,
            "init",
            "carry",
            include_contiguity=False,
        )

        xs_proxy = xs.as_proxy()
        init_proxy = init.as_proxy()
        additional_inputs_proxy = list(additional_inputs.as_proxy()) + list(
            combine_freevars_proxy
        )

        # Mutation handling: scan allows in-place writes only to
        # additional_inputs (lifted, loop-invariant tensors). Mutations of
        # init / xs are unsafe (see ScanOp.gen_schema for the contract) so
        # we graph-break here with a precise message. The remaining indices
        # are mapped through the parent-side input list to produce a
        # comma-joined ``mutated_arg_indices`` kwarg for the HOP call,
        # mirroring while_loop.
        n_init = len(init_vars)
        n_xs = len(xs_vars)

        disallowed = sorted(i for i in combine_mutated_inputs if i < n_init + n_xs)
        if disallowed:
            init_part = [i for i in disallowed if i < n_init]
            xs_part = [i - n_init for i in disallowed if n_init <= i]
            unimplemented(
                gb_type="torch.scan: combine_fn mutates init or xs",
                context=f"init={init_part}, xs={xs_part}",
                explanation=(
                    "scan only supports in-place mutation of additional_inputs "
                    "(loop-invariant tensors). init is the *initial* carry only -- "
                    "from step 1 on the driver feeds combine_fn's returned carry, "
                    "so an in-place write to init would only affect step 0. "
                    "xs[t] is a fresh, storage-disjoint slice each step, so a "
                    "mutation cannot be observed by step t+1. Update the carry "
                    "via the combine_fn return value; for in-place lifted buffers "
                    "use additional_inputs."
                ),
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # additional_inputs (subgraph indices [n_init+n_xs, ...)) plus any
        # lifted freevars (indices >= n_init+n_xs+n_additional). Both are
        # carried as additional_inputs at the HOP-call layer.
        all_scan_inputs: list[VariableTracker | Proxy] = (
            list(init_vars)
            + list(xs_vars)
            + list(additional_inputs_vars)
            + list(combine_freevars_proxy)
        )
        # Build a parent-storage set from the subgraph placeholders that were
        # detected as mutated, then re-map through ``all_scan_inputs`` so we
        # cover both regular additional_inputs and lifted freevars.
        mutated_input_storages = subgraph_mutated_input_storages(
            combine_graph,
            {i for i in combine_mutated_inputs if i >= n_init + n_xs},
        )
        mutated_inputs = parent_mutated_input_indices(
            all_scan_inputs, mutated_input_storages
        )
        mutated_arg_indices = ",".join(str(i) for i in mutated_inputs)

        combine_gm = torch.fx.GraphModule(dict(tx.output.nn_modules), combine_graph)
        combine_fn_name = tx.output.install_subgraph("scan_combine_fn", combine_gm)

        p_args = (
            make_attr(tx, combine_fn_name),
            init_proxy,
            xs_proxy,
            additional_inputs_proxy,
        )
        hop_kwargs = (
            {"mutated_arg_indices": mutated_arg_indices} if mutated_arg_indices else {}
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.scan,
            p_args,
            hop_kwargs,
            None,
            _combine_spec,
            None,
        )


class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for map_impl (mapping a function over the leading dimension of xs).
    # Hard-errors on graph breaks; input mutation support depends on grad mode.
    _HOP_NAME = "torch.ops.higher_order.map_impl"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces map(f, flat_xs, flat_args). Graph breaks on keyword args, a
        # non-callable f, or xs with scalar/empty leading dim. Speculates f on index-zero slices of
        # xs plus the extra args (flattened, constants kept), requires tensor (or none) outputs, and
        # graph breaks if f mutates the extra args or captured tensors (only xs mutation allowed;
        # mutated xs indices passed as mutated_arg_indices). Installs the body and emits the
        # map_impl call. Returns the unflattened output.
        self.supports_input_mutation = not torch.is_grad_enabled()

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if len(kwargs) > 0:
            unimplemented(
                gb_type="torch.map: kwargs not supported",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"torch.map expects no keyword arguments (got {len(kwargs)})",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        _check_supported_callable_arg(tx, args[0], "map_fn")

        # args = f, flat_xs, flat_args
        if not isinstance(args[1], (ListVariable, TupleVariable)):
            raise AssertionError(
                f"Expected args[1] to be a ListVariable or TupleVariable, got {args[1]}"
            )
        if not isinstance(args[2], (ListVariable, TupleVariable)):
            raise AssertionError(
                f"Expected args[2] to be a ListVariable or TupleVariable, got {args[2]}"
            )
        unpacked_xs = unpack_iterable(tx, args[1])
        unpacked_args = unpack_iterable(tx, args[2])

        sample_shape = get_fake_value(unpacked_xs[0].as_proxy().node, tx).size()

        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented(
                gb_type="torch.map: improper inputs",
                context=str(sample_shape),
                explanation="torch.map doesn't support scalar or non-zero sized tensors during tracing.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # To get the example output from map() we will need to provide at least one sample to
        # the loop body. In our case we will always use xs[0], and our map() won't support zero
        # sized tensor during tracing.
        with discard_graph_changes(tx):
            sliced_xs = [
                xs.call_method(
                    tx,
                    "select",
                    args=[VariableTracker.build(tx, 0), VariableTracker.build(tx, 0)],
                    kwargs={},
                )
                for xs in unpacked_xs
            ]
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")

        # TODO: Support kwargs
        (
            (body_r, body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            [
                *sliced_xs,
                *unpacked_args,
            ],
            {},
            self._HOP_NAME,
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            should_flatten_outputs=True,
            # TODO - removing consts from control flow ops need more work
            remove_consts_from_outputs=False,
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        body_mutated_inputs = set(
            getattr(body_graph, "_dynamo_mutated_input_indices", ())
        )

        # Check all outputs of map are tensors.
        # For map, outputting None is OK, thus ignore None values in the check
        body_r_vars = unpack_iterable(tx, body_r)
        none_mask = [x.is_constant_none() for x in body_r_vars]
        _check_all_tensorvariable(
            [br for bm, br in zip(none_mask, body_r_vars) if not bm]
        )

        # Mutation handling: map allows in-place writes only to xs
        # (each iteration sees a storage-disjoint slice). Mutations of
        # pos_args or captured freevars are unsafe (see MapImpl.gen_schema
        # for the contract). Placeholder order is [xs, pos_args, freevars].
        n_xs = len(unpacked_xs)
        n_pos = len(unpacked_args)

        pos_args_mutated = sorted(
            i - n_xs for i in body_mutated_inputs if n_xs <= i < n_xs + n_pos
        )
        if pos_args_mutated:
            unimplemented(
                gb_type="torch.map: f mutates pos_args",
                context=f"pos_args={pos_args_mutated}",
                explanation=(
                    "map only supports in-place mutation of xs (each "
                    "iteration sees a storage-disjoint slice). pos_args "
                    "are loop-invariant: every iteration sees the same "
                    "tensor, so a mutation makes iterations depend on each "
                    "other, breaking map's independence contract and "
                    "introducing a data race under any parallel lowering. "
                    "Use scan or while_loop if sequential buffer updates "
                    "are required."
                ),
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        freevars_mutated = sorted(
            i - n_xs - n_pos for i in body_mutated_inputs if i >= n_xs + n_pos
        )
        if freevars_mutated:
            unimplemented(
                gb_type="torch.map: f mutates a captured tensor",
                context=f"freevars={freevars_mutated}",
                explanation=(
                    "map only supports in-place mutation of xs. A tensor "
                    "captured from the enclosing scope (lifted parameter) is loop-invariant: "
                    "every iteration sees the same tensor, so a mutation "
                    "makes iterations depend on each other, breaking map's "
                    "independence contract and introducing a data race "
                    "under any parallel lowering. Pass the buffer through "
                    "xs, or use scan / while_loop if sequential buffer "
                    "updates are required."
                ),
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # No storage round-trip needed: aliasing graph-breaks upstream, so the
        # mutated subgraph placeholder indices are already the parent xs indices.
        mutated_arg_indices = ",".join(str(i) for i in sorted(body_mutated_inputs))

        body_nn_modules = dict(tx.output.nn_modules)

        body_name = tx.output.install_subgraph(
            "map_body",
            torch.fx.GraphModule(body_nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)

        p_args = (
            body_node,
            [xs.as_proxy() for xs in unpacked_xs],
            [arg.as_proxy() for arg in unpacked_args]
            + list(body_lifted_freevars.keys()),
        )
        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.map_impl,
            p_args,
            {"mutated_arg_indices": mutated_arg_indices} if mutated_arg_indices else {},
            None,
            body_spec,
            body_r,
        )


class PrintHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the print HOP; emits a graph call with proxied args and kwargs.
    _HOP_NAME = "torch.ops.higher_order.print"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Realizes args, proxies them, emits a call_function node for the print HOP
        # and returns the wrapped result (no example value).
        from .builder import wrap_fx_proxy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        args_proxy = [arg.as_proxy() for arg in args]
        kwargs_proxy = {k: v.as_proxy() for k, v in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(args_proxy),
                kwargs=kwargs_proxy,
            ),
        )


class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the Executorch delegate-call HOP: the first arg is a lowered module
    # (installed in the graph), remaining args are proxied; the example output is computed by
    # running the lowered module's original module on fake inputs.
    _HOP_NAME = "torch.ops.higher_order.executorch_call_delegate"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Graph breaks on kwargs or when the first arg is not an nn.Module variable.
        # Resolves the lowered module (registering it as a graph attribute), evaluates its original
        # module on fake values under the fake mode to get the example output, asserts outputs do
        # not alias inputs (AssertionError), then emits the call and returns the wrapped result.
        from .builder import wrap_fx_proxy

        # This is operator for delegation within Executorch which calls a
        # specific function in the given lowered module with the given
        # operators. The actual operator is defined in the Executorch codebase.
        # This is a bad hierarchical violation since
        # executorch_call_delegate sits at a higher level than dynamo, but
        # there's no real solution to this issue yet.
        if len(kwargs) > 0:
            unimplemented(
                gb_type="executorch_call_delegate: kwargs not supported",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"executorch_call_delegate expects no keyword arguments (got {len(kwargs)})",
                hints=[],
            )
        lowered_module, lowered_node = None, None
        if isinstance(args[0], variables.NNModuleVariable):
            lowered_module = tx.output.get_submodule(args[0].module_key)
            lowered_node = make_attr(tx, args[0].module_key)
        elif isinstance(args[0], variables.UnspecializedNNModuleVariable):
            # This nn module is special sa delegated by executorch. Just
            # install it as a attr in the graph.
            lowered_module = args[0].value
            lowered_node = tx.output.register_static_attr_and_return_proxy(
                "delegate", lowered_module
            )
        else:
            unimplemented(
                gb_type="executorch_call_delegate: first arg not supported",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"executorch_call_delegate expects the first argument to be a nn.Module (got {args[0]})",
                hints=[],
            )

        p_args = tuple(arg.as_proxy() for arg in args[1:])
        real_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: get_fake_value(a.node, tx), p_args
        )
        if tx.fake_mode is None:
            raise AssertionError("tx.fake_mode must not be None")
        with tx.fake_mode:
            example_value = lowered_module.original_module.module()(*real_sub_args)  # type: ignore[attr-defined]

        # NOTE [Guaranteeing the 1-1 correspondence of FakeTensors and real tensors]:
        # executorch modules promise not to alias inputs and outputs.
        # Thus, output FakeTensors will correctly not alias input FakeTensors.
        _assert_tensors_nonaliasing(real_sub_args, example_value)

        p_args = (lowered_node,) + p_args

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


class FunctorchHigherOrderVariable(UserFunctionVariable):
    # [CLASS DOC] User function variable for functorch-style transforms that are inlined like
    # ordinary functions but forbid nested graph breaks.
    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Pass-through to the parent call_function (inlines the function).
        return super().call_function(tx, args, kwargs)

    def should_allow_nested_graph_breaks(self) -> bool:
        # [FUNCTION DOC] Always returns false: graph breaks inside are not allowed to nest.
        return False


class FunctionalCallVariable(FunctorchHigherOrderVariable):
    # [CLASS DOC] Functorch variable for the functional_call API; behavior is inherited unchanged.
    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Pass-through to the parent call_function.
        return super().call_function(tx, args, kwargs)


class ReparametrizeModuleCallVariable(FunctorchHigherOrderVariable):
    # [CLASS DOC] Functorch variable for the module reparametrization call: result is wrapped as
    # a reparametrize-module context manager variable bound to the first argument.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Forwards all args to the parent constructor.
        super().__init__(*args, **kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Calls the parent to get a context manager variable, then wraps it with
        # the module (first arg) in a reparametrize-module context variable and returns that.
        ctx_manager_vt = super().call_function(tx, args, kwargs)
        return RepararametrizeModuleContextVariable(ctx_manager_vt, args[0])  # type: ignore[arg-type]


class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the wrap HOP and base for other "wrapper" HOPs (flex_gemm, hints
    # wrapper, checkpoint, dynamo bypassing wrapper). Class flags control mutation, aliasing and
    # side-effect support. create_wrapped_node speculates the body with automatic output
    # flattening and installs it as a subgraph.
    _HOP_NAME = "torch.ops.higher_order.wrap"
    supports_input_mutation = True
    supports_aliasing = True
    allow_side_effects = False

    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslatorBase",
        fn_vt: VariableTracker,
        fn_args_vt: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
        body_gmod: GraphModule,
        attr_name: str = "wrap_body",
    ) -> str:
        # [FUNCTION DOC] Installs the body graph module in the output graph under the given
        # attribute name and returns the name actually assigned. Overridable hook.
        return tx.output.install_subgraph(
            f"{attr_name}",
            body_gmod,
        )

    def create_wrapped_node(
        self,
        tx: "InstructionTranslatorBase",
        fn_vt: VariableTracker,
        fn_args_vt: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
        description: str,
        *,
        subgraph_name: str = "wrap_body",
        set_subgraph_inputs: Literal[
            "automatic", "automatic_with_forced_inputs"
        ] = "automatic",
    ) -> tuple[
        tuple[Proxy, ...],
        dict[str, VariableTracker],
        Any,
        VariableTracker,
        GraphModule,
        str,
        VariableTracker | tuple[VariableTracker, ...],
        SubgraphTracingInfo,
    ]:
        # [FUNCTION DOC] Speculates fn_vt on fn_args_vt/kwargs with auto output flattening (using
        # this class's side-effect/mutation/aliasing flags), builds a graph module, installs it,
        # and builds the proxy args for the HOP call: body node followed by lifted free variables
        # (in forced-input mode, the tensor args first). Returns (proxy args, empty kwargs,
        # example value of the output node, output variable, graph module, subgraph name, graph
        # output variables, tracing info).
        # See NOTE [HigherOrderOperator tracing design] for more details
        (
            body_r,
            body_graph,
            body_lifted_freevars,
            body_graph_output_vts,
            tracing_info,
        ) = speculate_subgraph_with_auto_output_flattening(
            tx,
            fn_vt,
            fn_args_vt,
            kwargs,
            description,
            source_target=self.value,
            set_subgraph_inputs=set_subgraph_inputs,
            allow_side_effects=self.allow_side_effects,
            filter_aliased_intermediates=getattr(
                self, "filter_aliased_intermediates", False
            ),
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = self.install_subgraph_in_output_graph(
            tx,
            fn_vt,
            fn_args_vt,
            kwargs,
            body_gmod,
            attr_name=subgraph_name,
        )
        body_node = make_attr(tx, body_name)

        if set_subgraph_inputs == "automatic_with_forced_inputs":
            # Forced mode creates placeholders eagerly (in user arg order) and
            # rewraps the tensor VTs with child-tracer proxies, so these args
            # never go through the freevar lift path. Build the outer p_args
            # from fn_args_vt directly, then append any closure captures that
            # were auto-lifted during body tracing.
            forced_args = tuple(a.as_proxy() for a in fn_args_vt if a.is_tensor())
            lifted_args = forced_args + tuple(body_lifted_freevars)
        else:
            # With "automatic", all args flow through the freevar lift path.
            lifted_args = tuple(body_lifted_freevars)

        proxy_args = (body_node,) + lifted_args

        example_value = pytree.tree_map_only(
            torch.fx.Node,
            lambda a: a.meta["example_value"],
            body_graph.find_nodes(op="output")[0].args[0],
        )

        return (
            proxy_args,
            {},
            example_value,
            body_r,
            body_gmod,
            body_name,
            body_graph_output_vts,
            tracing_info,
        )

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces wrap(fn, *args): builds the wrapped body node, graph breaks if any
        # kwargs remain unflattened, then emits the HOP call and reproxifies outputs. Returns the
        # output variable.
        # This flattens the kwargs into lifted args
        (
            p_args,
            p_kwargs,
            _example_value,
            body_r,
            _,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(tx, args[0], args[1:], kwargs, "wrap")

        if len(p_kwargs) > 0:
            unimplemented(
                gb_type="WrapHigherOrderVariable: kwargs unexpected",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="kwargs should have been flattened into lifted args.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            tuple(p_args),
            p_kwargs,
            _example_value,
            body_r,
            body_graph_output_vts,
        )


class WrapWithSetGradEnabledHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for wrap_with_set_grad_enabled (post-export HOP): traces a body under a
    # constant grad-enabled flag into a subgraph that may not have free variables.
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """

    _HOP_NAME = "torch.ops.higher_order.wrap_with_set_grad_enabled"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Overrides call_function directly. Takes (grad_enabled constant, fn,
        # *operands); graph breaks on kwargs, non-constant flag, non-callable fn, or if the body
        # lifts free variables. Speculates fn with manual inputs under the grad mode set via the
        # global setter (restored afterwards), installs the subgraph and emits the HOP call.
        # Returns the unflattened output.
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if kwargs:
            unimplemented(
                gb_type="wrap_with_set_grad_enabled: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"wrap_with_set_grad_enabled expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        grad_enabled, fn_var, *rest_args = args

        if not grad_enabled.is_python_constant():
            unimplemented(
                gb_type="wrap_with_set_grad_enabled: non-constant grad_enabled",
                context=str(grad_enabled),
                explanation="wrap_with_set_grad_enabled expects grad_enabled argument to be a constant.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        _check_supported_callable_arg(tx, fn_var, "enable_grad_fn")

        with torch.set_grad_enabled(grad_enabled.as_python_constant()):
            if self._HOP_NAME is None:
                raise AssertionError("_HOP_NAME must be set")
            (
                (body_r, treespec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn_var,
                [*rest_args],
                {},
                self._HOP_NAME,
                source_target=self.value,
                set_subgraph_inputs="manual",
                should_flatten_outputs=True,
            )

        if len(body_lifted_freevars) > 0:
            unimplemented(
                gb_type="wrap_with_set_grad_enabled: unexpected freevars",
                context=str(body_lifted_freevars),
                explanation="wrap_with_set_grad_enabled expects no freevars.",
                hints=[],
            )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        proxy_args = tuple(
            [
                grad_enabled.as_python_constant(),
                body_node,
            ]
            + [operand.as_proxy() for operand in rest_args]
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )
        return _call_function_and_unflatten_output(
            tx, self.value, proxy_args, {}, example_value, treespec, body_r
        )


class WrapWithAutocastHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for wrap_with_autocast (post-export HOP): traces a body under an
    # autocast context with constant settings into a subgraph without free variables.
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """

    _HOP_NAME = "torch.ops.higher_order.wrap_with_autocast"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Overrides call_function directly. Takes (device_type, dtype, enabled,
        # cache_enabled, fn, *operands); graph breaks on kwargs, non-constant autocast settings,
        # non-callable fn, or lifted free variables. Speculates fn with manual inputs inside a real
        # autocast context, installs the subgraph and emits the HOP call with the constants.
        # Returns the unflattened output.
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if kwargs:
            unimplemented(
                gb_type="wrap_with_autocast: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"wrap_with_autocast expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        device_type, dtype, enabled, cache_enabled, fn_var, *rest_args = args

        for arg in [device_type, dtype, enabled, cache_enabled]:
            if not arg.is_python_constant():
                unimplemented(
                    gb_type="wrap_with_autocast: expected constant arg",
                    context=str(args),
                    explanation="wrap_with_autocast expects device_type, dtype, enabled, "
                    "and cache_enabled arguments to be constants.",
                    hints=[
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )

        _check_supported_callable_arg(tx, fn_var, "autocast")

        python_constants = [
            arg.as_python_constant()
            for arg in [device_type, dtype, enabled, cache_enabled]
        ]

        with torch.autocast(*python_constants):
            if self._HOP_NAME is None:
                raise AssertionError("_HOP_NAME must be set")
            (
                (body_r, treespec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn_var,
                [*rest_args],
                {},
                self._HOP_NAME,
                source_target=self.value,
                set_subgraph_inputs="manual",
                should_flatten_outputs=True,
            )

        if len(body_lifted_freevars) > 0:
            unimplemented(
                gb_type="wrap_with_autocast: unexpected freevars",
                context=str(body_lifted_freevars),
                explanation="wrap_with_autocast expects no freevars.",
                hints=[],
            )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        proxy_args = tuple(
            [
                *python_constants,
                body_node,
            ]
            + [operand.as_proxy() for operand in rest_args]
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx, self.value, proxy_args, {}, example_value, treespec, body_r
        )


class FlexGemmHigherOrderVariable(WrapHigherOrderVariable):
    # [CLASS DOC] Variable for the internal flex_gemm HOP (gemm op with a user body function).
    # Hard-errors on graph breaks; reuses the wrap machinery to trace the body.
    _HOP_NAME = "torch.ops.higher_order.flex_gemm"
    _ALLOW_FALLBACK_TO_EAGER = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Expects exactly five positional args (gemm_op, body_fn, operands list,
        # kwargs dict, kernel_options dict) and no keywords; graph breaks otherwise. Traces the body
        # via create_wrapped_node, maps the gemm op through an alias table, and emits the call with
        # (op, body node, lifted operands, kwargs, kernel options). Returns the output variable.
        if kwargs:
            unimplemented(
                gb_type="flex_gemm: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="flex_gemm internal HOP expects positional args only.",
                hints=[*graph_break_hints.USER_ERROR],
            )

        if (
            len(args) != 5
            or not isinstance(args[2], (ListVariable, TupleVariable))
            or not isinstance(args[3], ConstDictVariable)
            or not isinstance(args[4], ConstDictVariable)
        ):
            unimplemented(
                gb_type="flex_gemm: improper args/kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=(
                    "flex_gemm expects 5 positional arguments: "
                    "gemm_op, body_fn, args, kwargs, kernel_options. args is "
                    "expected to be list/tuple and kwargs/kernel_options are "
                    "expected to be dicts."
                ),
                hints=[*graph_break_hints.USER_ERROR],
            )

        _check_supported_callable_arg(tx, args[1], "body_fn")
        operands = args[2].unpack_var_sequence(tx)
        fn_kwargs = args[3].as_python_constant()
        kernel_options = args[4].as_python_constant()
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        (
            p_args,
            _,
            example_value,
            body_r,
            _,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[1],
            operands,
            {},
            self._HOP_NAME,
            subgraph_name="flex_gemm_body",
        )

        body_node = p_args[0]
        gemm_op = args[0].as_python_constant()
        p_args = (
            FLEX_GEMM_OP_ALIASES.get(gemm_op, gemm_op),
            body_node,
            tuple(p_args[1:]),
            fn_kwargs,
            kernel_options,
        )

        return cast(
            VariableTracker,
            _call_function_with_auto_output_flattening(
                tx,
                self.value,
                p_args,
                {},
                example_value,
                body_r,
                body_graph_output_vts,
            ),
        )


class HintsWrapperHigherOrderVariable(WrapHigherOrderVariable):
    # [CLASS DOC] Variable for hints_wrapper(body_fn, args, kwargs, hints=...). Hard-errors on
    # graph breaks; the body is installed under its own subgraph name.
    _HOP_NAME = "torch.ops.higher_order.hints_wrapper"
    _ALLOW_FALLBACK_TO_EAGER = False

    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslatorBase",
        fn_vt: VariableTracker,
        fn_args_vt: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
        body_gmod: GraphModule,
        attr_name: str = "wrap_body",
    ) -> str:
        # [FUNCTION DOC] Installs the body under the fixed name for this HOP, ignoring attr_name.
        # Returns the assigned name.
        return tx.output.install_subgraph(
            "hints_wrapper_body",
            body_gmod,
        )

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Requires a callable body_fn, a list/tuple of args, a dict of kwargs and
        # exactly one keyword argument named hints; graph breaks otherwise. Traces the body with the
        # kwargs dict, restructures the proxy args as (body node, tuple of lifted args, empty dict),
        # passes the hints as a constant keyword to the call, and returns the output variable.
        _check_supported_callable_arg(tx, args[0], "body_fn")

        # inputs
        if (
            len(args) != 3
            or not isinstance(args[1], (ListVariable, TupleVariable))
            or not isinstance(args[2], ConstDictVariable)
            or len(kwargs) != 1
            or "hints" not in kwargs
        ):
            unimplemented(
                gb_type="hints_wrapper: improper args/kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"hints_wrapper expects 3 positional arguments (got {len(args)}) "
                f"and 1 keyword argument (got {len(kwargs)}). "
                "Usage: hints_wrapper(body_fn, args, kwargs, hints=...). "
                "args is expected to be list/tuple and kwargs is expected to be a dict.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        operands = unpack_iterable(tx, args[1])
        fn_kwargs = args[2].as_python_constant()
        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        # Use create_wrapped_node from WrapHigherOrderVariable
        (
            p_args,
            _,
            example_value,
            body_r,
            body_gmod,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[0],  # function
            operands,
            fn_kwargs,
            self._HOP_NAME,
        )

        # hints_wrapper expects (body_node, args, kwargs) as positional args
        # So we need to restructure p_args from (body_node, *lifted_args)
        # to (body_node, lifted_args_tuple, {})
        body_node = p_args[0]
        lifted_args = p_args[1:]
        # pyrefly: ignore [implicit-any]
        p_args = (body_node, tuple(lifted_args), {})

        # add hints into p_kwargs
        p_kwargs = {}
        p_kwargs["hints"] = kwargs["hints"].as_python_constant()

        return _call_function_with_auto_output_flattening(  # type: ignore[return-type]
            tx,
            self.value,
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
        )


class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for out_dtype(op, output_dtype, *args); computes the example value by
    # running the op on example inputs and casting to the output dtype.
    _HOP_NAME = "torch.ops.higher_order.out_dtype"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Graph breaks on kwargs. Proxies the args, computes a simplified example
        # value (op applied to example inputs, cast to the output dtype, without argument
        # promotion), emits the call and returns the wrapped result.
        from .builder import wrap_fx_proxy

        if len(kwargs) > 0:
            unimplemented(
                gb_type="out_dtype: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"out_dtype expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        p_args = tuple(arg.as_proxy() for arg in args)
        op = p_args[0]
        output_dtype = p_args[1]
        fake_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: a.node.meta["example_value"], p_args[2:]
        )
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        example_value = op(*fake_sub_args).to(dtype=output_dtype)

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


class StrictModeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for strict_mode(body, operands). Hard-errors on graph breaks.
    _HOP_NAME = "torch.ops.higher_order.strict_mode"
    _ALLOW_FALLBACK_TO_EAGER = False

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Unpacks the operands (second arg); graph breaks if any operand is itself a
        # container or if kwargs are given. Speculates the body with flattened outputs, installs it,
        # and emits the call with (body node, lifted free variables). Returns the unflattened
        # output.
        unpacked_sequence = unpack_iterable(tx, args[1])
        # TODO (tmanlaibaatar) support pytree here
        for arg in unpacked_sequence:
            if isinstance(
                arg, (ListVariable, TupleVariable, ConstDictVariable, SetVariable)
            ):
                unimplemented(
                    gb_type="strict_mode: improper args",
                    context=f"args: {args}, kwargs: {kwargs}",
                    explanation="strict_mode higher order op expects flat inputs (list/tuple/dict/set)",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

        if kwargs:
            unimplemented(
                gb_type="strict_mode: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"strict_mode higher order op expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must be set")
        (
            (ret_val, ret_spec),
            ret_graph,
            ret_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            unpacked_sequence,
            {},
            self._HOP_NAME,
            source_target=self.value,
            should_flatten_outputs=True,
        )

        strict_mode_nn_modules = dict(tx.output.nn_modules)

        strict_mode_name = tx.output.install_subgraph(
            "strict_mode_body",
            torch.fx.GraphModule(strict_mode_nn_modules, ret_graph),
        )

        strict_mode_node = make_attr(tx, strict_mode_name)
        p_args = (
            strict_mode_node,
            tuple(ret_lifted_freevars.keys()),
        )

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            ret_val.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.strict_mode,
            p_args,
            {},
            flat_example_value,
            ret_spec,
            ret_val,
        )


class CheckpointHigherOrderVariable(WrapHigherOrderVariable):
    # [CLASS DOC] Variable for torch.utils.checkpoint.checkpoint (activation checkpointing). Side
    # effects inside the region are allowed according to a dynamo config flag read at
    # construction. Traces the function as a wrapped subgraph and passes checkpoint options as
    # call kwargs.
    _HOP_NAME = "torch.utils.checkpoint.checkpoint"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Initializes the parent and sets allow_side_effects from the dynamo config
        # skip_fwd_side_effects_in_bwd_under_checkpoint.
        super().__init__(*args, **kwargs)
        self.allow_side_effects = (
            torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint
        )

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Extracts a custom context_fn from kwargs when it is not the default
        # (user function or partial; other kinds raise NotImplementedError), splits kwargs into
        # checkpoint options and function kwargs, traces the function via create_wrapped_node,
        # stores the context function in the checkpointed graph module metadata, and emits the
        # HOP call with the checkpoint kwargs proxied. Returns the output variable.
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn

        context_fn = None
        if "context_fn" in kwargs and kwargs["context_fn"] is not noop_context_fn:
            ctx = kwargs.pop("context_fn")
            if isinstance(ctx, torch._dynamo.variables.UserFunctionVariable):
                context_fn = ctx.fn
            elif isinstance(
                ctx, torch._dynamo.variables.functions.FunctoolsPartialVariable
            ):
                context_fn = ctx.guard_as_python_constant()
            else:
                raise NotImplementedError(
                    f"checkpoint not implemented for {type(ctx)} context_fn"
                )

        checkpoint_kwargs, gmod_kwargs = TagActivationCheckpoint.divide_kwargs(kwargs)

        # Here we use checkpoint_kwargs (and not gmod kwargs). gmod_kwargs are
        # already flattened above and managed inside the fx graph.
        (
            p_args,
            _,
            example_value,
            _body_r,
            checkpointed_gmod,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[0],
            args[1:],
            gmod_kwargs,
            "torch.utils.checkpoint.checkpoint",
        )
        if context_fn is not None:
            checkpointed_gmod.meta["_checkpoint_context_fn"] = context_fn

        _, checkpoint_kwargs = proxy_args_kwargs([], checkpoint_kwargs)

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            p_args,
            checkpoint_kwargs,
            example_value,
            _body_r,
            body_graph_output_vts,
        )


class DynamoBypassingWrapperHigherOrderVariable(WrapHigherOrderVariable):
    # [CLASS DOC] Variable for dynamo_bypassing_wrapper(wrapper_fn, fn, *args): traces fn as a
    # wrapped subgraph and attaches the (untraced) wrapper function to the subgraph metadata.
    _HOP_NAME = "torch.ops.higher_order.dynamo_bypassing_wrapper"

    def __init__(self, hop: HigherOrderOperator, source: Source | None) -> None:
        # [FUNCTION DOC] HELPER: Forwards the HOP and source to the parent constructor.
        super().__init__(hop, source)

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Resolves the first arg to a real python function (user function or
        # partial; otherwise raises RuntimeError), traces the second arg with the remaining args
        # via create_wrapped_node, stores the wrapper function in the graph module metadata under a
        # fixed key, and emits the call whose first argument is that key. Returns the output
        # variable.
        func_var = args[0]

        if isinstance(func_var, torch._dynamo.variables.UserFunctionVariable):
            func = func_var.fn
        elif isinstance(
            func_var, torch._dynamo.variables.functions.FunctoolsPartialVariable
        ):
            func = func_var.as_python_constant()
        else:
            raise RuntimeError(
                f"DynamoBypassingWrapperHigherOrderVariable: Unsupported function {type(func_var)}"
            )
        (
            p_args,
            _,
            example_value,
            _body_r,
            gmod,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[1],
            args[2:],
            kwargs,
            str(func),
        )

        # Alternatively, we could've stored only the function's fqn and
        # reconstructed, but that requires the function to be a global.
        gmod_meta_key = "_dynamo_bypassing_wrapper_fn"
        gmod.meta[gmod_meta_key] = func

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            (gmod_meta_key,) + tuple(p_args),
            {},
            example_value,
            _body_r,
            body_graph_output_vts,
        )


class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the export tracepoint HOP; just proxies args and emits the call.
    _HOP_NAME = "torch.ops.higher_order._export_tracepoint"

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Overrides call_function directly: proxies args/kwargs, emits a
        # call_function node and returns the wrapped result with no example value.
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class RunWithRNGStateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for run_with_rng_state; proxies args and emits the call.
    _HOP_NAME = "torch.ops.higher_order.run_with_rng_state"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Proxies args/kwargs, emits a call_function node and returns the wrapped
        # result with no example value.
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class InlineAsmElementwiseHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the inline_asm_elementwise HOP; proxies args and emits the call.
    _HOP_NAME = "inline_asm_elementwise"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Proxies args/kwargs, emits a call_function node and returns the wrapped
        # result with no example value.
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class AutoFunctionalizeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for auto_functionalized; proxies args and emits the call.
    _HOP_NAME = "torch.ops.higher_order.auto_functionalized"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Proxies args/kwargs, emits a call_function node and returns the wrapped
        # result with no example value.
        from .builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class FlexAttentionBackwardHighOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for flex_attention_backward. Handles calls that already carry
    # pre-traced forward/joint graph modules (14 args) or falls back to tracing user functions.
    # Helpers trace score/mask functions into subgraphs and derive the joint graph.
    _HOP_NAME = "torch.ops.higher_order.flex_attention_backward"

    @staticmethod
    def _uses_pretraced_graphs(
        fw_graph: VariableTracker, joint_graph: VariableTracker
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if a joint graph is supplied or the forward graph is an
        # unspecialized module variable (graphs were already traced).
        return not joint_graph.is_constant_none() or isinstance(
            fw_graph, UnspecializedNNModuleVariable
        )

    def proxy_submod(
        self, tx: "InstructionTranslatorBase", arg: UnspecializedNNModuleVariable
    ) -> Proxy:
        # [FUNCTION DOC] Installs a pre-traced module variable as a subgraph attribute named by
        # the dictionary key in its source, records its example value, and returns the get_attr
        # proxy. Raises AssertionError if the source is missing or not rooted at a dict item.
        if not arg.source:  # type: ignore[attr-defined]
            raise AssertionError("arg.source must be set")
        if not isinstance(arg.source.base, DictGetItemSource):  # type: ignore[attr-defined]
            raise AssertionError("arg.source.base must be a DictGetItemSource")
        submod_name = tx.output.install_subgraph(arg.source.base.index, arg.value)  # type: ignore[arg-type]
        p_submod = make_attr(tx, submod_name)
        set_example_value(p_submod.node, arg.value)
        return p_submod

    def to_proxy(self, tx: "InstructionTranslatorBase", arg: VariableTracker) -> Any:
        # [FUNCTION DOC] Converts a variable to a proxy: module variables are installed as
        # subgraphs, lists/tuples are converted element-wise into the same container type, others
        # use as_proxy.
        if isinstance(arg, UnspecializedNNModuleVariable):
            return self.proxy_submod(tx, arg)
        elif isinstance(arg, (ListVariable, TupleVariable)):
            return arg.python_type()(
                self.to_proxy(tx, nested_arg) for nested_arg in arg.items
            )
        else:
            return arg.as_proxy()

    def create_wrapped_node(
        self,
        tx: "InstructionTranslatorBase",
        query: VariableTracker,
        fn: VariableTracker,
        fn_name: str,
        other_buffers: list[VariableTracker],
    ) -> tuple[Proxy, tuple[Proxy, ...], torch.fx.GraphModule]:
        # [FUNCTION DOC] Traces the user score_mod or mask_fn into a subgraph. Builds sample scalar
        # inputs (four int32 scalars for batch/head/q/kv, plus a score scalar for score_mod) under
        # discarded graph changes, then speculates fn with item indexing transformed, installs the
        # graph module and returns (get_attr proxy, lifted free variable proxies, graph module).
        # Raises AssertionError for a function name other than score_mod or mask_fn.
        from .._trace_wrapped_higher_order_op import TransformGetItemToIndex

        def create_scalar() -> VariableTracker:
            # [FUNCTION DOC] HELPER: Creates an empty int32 scalar tensor like the query.
            return query.call_method(
                tx,
                "new_empty",
                [VariableTracker.build(tx, [])],
                {"dtype": VariableTracker.build(tx, torch.int32)},
            )

        with discard_graph_changes(tx):
            bhmn = [create_scalar() for _ in range(4)]
            if fn_name == "score_mod":
                scores_require_grad: bool = query.requires_grad  # type: ignore[attr-defined]
                score = query.call_method(
                    tx,
                    "new_empty",
                    [VariableTracker.build(tx, [])],
                    {"requires_grad": VariableTracker.build(tx, scores_require_grad)},
                )
                new_args = [score, *bhmn, *other_buffers]
            else:
                if fn_name != "mask_fn":
                    raise AssertionError("Illegal function name: " + fn_name)
                new_args = [*bhmn, *other_buffers]

        with TransformGetItemToIndex():
            (
                (_body_output, _body_spec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn,
                new_args,
                {},
                description=f"{self._HOP_NAME}: {fn_name}",
                source_target=self.value,
                set_subgraph_inputs="flatten_manual",
            )

        gm = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(fn_name, gm)
        return make_attr(tx, body_name), tuple(body_lifted_freevars), gm

    @staticmethod
    def _buffer_example_value(buffer: VariableTracker) -> Any:
        # [FUNCTION DOC] HELPER: Returns the example value of the buffer's proxy node, or the raw
        # value if it is not a proxy.
        proxy = buffer.as_proxy()
        if isinstance(proxy, Proxy):
            return proxy.node.meta["example_value"]
        return proxy

    def _derive_joint_graph(
        self,
        tx: "InstructionTranslatorBase",
        query: VariableTracker,
        fw_graph_gm: torch.fx.GraphModule,
        score_mod_other_buffers: TupleVariable,
        fw_graph_lifted_args: tuple[Proxy, ...],
    ) -> Proxy:
        # [FUNCTION DOC] HELPER: Builds example inputs from the query (a grad-requiring score
        # scalar and four int scalars) plus buffer example values, derives the joint
        # forward/backward graph from the forward graph module, installs it as a subgraph with
        # its example value, and returns its get_attr proxy.
        from torch._higher_order_ops.flex_attention import create_fw_bw_graph

        query_example = query.as_proxy().node.meta["example_value"]
        example_vals = (
            query_example.new_zeros((), requires_grad=True),
            query_example.new_zeros((), dtype=torch.int),
            query_example.new_zeros((), dtype=torch.int),
            query_example.new_zeros((), dtype=torch.int),
            query_example.new_zeros((), dtype=torch.int),
        )
        all_buffer_examples = tuple(
            self._buffer_example_value(buf) for buf in score_mod_other_buffers.items
        ) + tuple(buf.node.meta["example_value"] for buf in fw_graph_lifted_args)

        _, joint_gm = create_fw_bw_graph(fw_graph_gm, example_vals, all_buffer_examples)
        joint_gm = cast(GraphModule, joint_gm)

        submod_name = tx.output.install_subgraph("joint_graph", joint_gm)
        p_submod = make_attr(tx, submod_name)
        set_example_value(p_submod.node, joint_gm)
        return p_submod

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces flex_attention_backward. If the call does not have the expected 14
        # args, has non-tuple block mask / buffers, or already carries pre-traced graphs, uses the
        # generic fallback. Otherwise traces score_mod into a forward graph, derives a joint graph,
        # traces the mask function (default no-op mask when none), and emits the call with the
        # lifted buffers appended; the original aten op is set as context. Returns the wrapped
        # result with no example value.
        from .builder import wrap_fx_proxy

        if len(args) != 14:
            return self._call_function_fallback(tx, args, kwargs)

        (
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ) = args

        if (
            not isinstance(block_mask, TupleVariable)
            or not isinstance(score_mod_other_buffers, TupleVariable)
            or not isinstance(mask_mod_other_buffers, TupleVariable)
            or len(block_mask.items) < 1
        ):
            return self._call_function_fallback(tx, args, kwargs)

        if self._uses_pretraced_graphs(fw_graph, joint_graph):
            return self._call_function_fallback(tx, args, kwargs)

        fw_graph_node, fw_graph_lifted_args, fw_graph_gm = self.create_wrapped_node(
            tx, query, fw_graph, "score_mod", score_mod_other_buffers.items
        )

        joint_graph_node = self._derive_joint_graph(
            tx,
            query,
            fw_graph_gm,
            score_mod_other_buffers,
            fw_graph_lifted_args,
        )

        mask_fn = block_mask.items[-1]
        if mask_fn.is_python_constant() and mask_fn.as_python_constant() is None:
            mask_fn = VariableTracker.build(
                tx,
                torch.nn.attention.flex_attention.noop_mask,
                source=mask_fn.source,
            )
        mask_fn_node, mask_fn_lifted_args, _ = self.create_wrapped_node(
            tx, query, mask_fn, "mask_fn", mask_mod_other_buffers.items
        )

        proxied_args = [
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            TupleVariable(block_mask.items[:-1], source=block_mask.source),
            scale,
            kernel_options,
        ]
        inp_args, _ = proxy_args_kwargs(proxied_args, {})
        proxied_score_mod_other_buffers = tuple(
            self.to_proxy(tx, arg) for arg in score_mod_other_buffers.items
        )
        proxied_mask_mod_other_buffers = tuple(
            self.to_proxy(tx, arg) for arg in mask_mod_other_buffers.items
        )

        (
            inp_q,
            inp_k,
            inp_v,
            inp_out,
            inp_lse,
            inp_grad_out,
            inp_grad_lse,
            inp_block_mask,
            inp_scale,
            inp_kernel_options,
        ) = inp_args

        block_mask_proxy = tuple(inp_block_mask + (mask_fn_node,))

        with torch.fx.experimental.proxy_tensor.set_original_aten_op(self.value):
            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    args=(
                        inp_q,
                        inp_k,
                        inp_v,
                        inp_out,
                        inp_lse,
                        inp_grad_out,
                        inp_grad_lse,
                        fw_graph_node,
                        joint_graph_node,
                        block_mask_proxy,
                        inp_scale,
                        inp_kernel_options,
                        proxied_score_mod_other_buffers + fw_graph_lifted_args,
                        proxied_mask_mod_other_buffers + mask_fn_lifted_args,
                    ),
                    kwargs={},
                ),
                example_value=None,
            )

    def _call_function_fallback(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Generic path: converts all args/kwargs to proxies (installing
        # module variables as subgraphs) and emits the call with no example value. Graph breaks if
        # an argument cannot be converted.
        from .builder import wrap_fx_proxy

        p_args, p_kwargs = None, None
        try:
            p_args = tuple(self.to_proxy(tx, arg) for arg in args)
            p_kwargs = {key: self.to_proxy(tx, arg) for key, arg in kwargs.items()}
        except (NotImplementedError, Unsupported) as err:
            unimplemented(
                gb_type="failed to handle argument for FlexAttentionBackward HOP",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="Missing Dynamo support for FlexAttentionBackward HOP argument.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
                from_exc=err,
            )
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the trace-wrapped inner_trace HOP; unwraps it by inlining the
    # function passed as the fn keyword.
    """
    Handles torch._dynamo._trace_wrapped_higher_order_op.inner_trace
    by unwrapping the higher order op and inlining through it.  This op
    is created by dynamo to survive through AotAutograd, then unwrapped
    here in the call to dynamo from compiled autograd.
    """

    _HOP_NAME = "torch._dynamo._trace_wrapped_higher_order_op.inner_trace"

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Copies kwargs, removes the fn entry (a missing fn raises a key error) and
        # calls it with the positional args and remaining kwargs, inlining it. Returns its result.
        kwargs = dict(kwargs)
        fn = kwargs.pop("fn")
        return fn.call_function(tx, args, kwargs)


class FlexAttentionHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # [CLASS DOC] Variable for the flex_attention HOP: traces the user score_mod and mask function
    # into subgraphs and emits the HOP call with them and their lifted free variables.
    _HOP_NAME = "torch.ops.higher_order.flex_attention"

    @staticmethod
    def normalize_to_args(
        args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> list[VariableTracker]:
        # [FUNCTION DOC] Returns the positional args followed by the flattened keyword values
        # (block mask tuple is not flattened) so the call can be unpacked positionally.
        # input signature is (query, key, value, score_mod, block_mask, *other_buffers),
        # block_mask is a tuple, and we don't want to flatten it.
        # only flatten kwargs into lists
        flat_kwargs = pytree.tree_flatten(kwargs)[0]

        # Combine the flattened lists
        all_args = args + flat_kwargs
        return all_args

    def create_wrapped_node(
        self,
        tx: "InstructionTranslatorBase",
        query: VariableTracker,
        fn: VariableTracker,
        fn_name: str,
    ) -> tuple[Proxy, tuple[Proxy, ...]]:
        # [FUNCTION DOC] Traces score_mod or mask_fn into a subgraph using sample int32 scalars
        # (plus a score scalar for score_mod) created under discarded graph changes, installs it,
        # and returns (get_attr proxy of the body, tuple of lifted free variable proxies). Raises
        # AssertionError for an unknown function name.
        from .._trace_wrapped_higher_order_op import TransformGetItemToIndex

        def create_scalar() -> VariableTracker:
            # [FUNCTION DOC] HELPER: Creates an empty int32 scalar tensor like the query.
            return query.call_method(
                tx,
                "new_empty",
                [
                    VariableTracker.build(tx, []),
                ],
                {
                    "dtype": VariableTracker.build(tx, torch.int32),
                },
            )

        with discard_graph_changes(tx):
            bhmn = [create_scalar() for _ in range(4)]
            if fn_name == "score_mod":
                scores_require_grad: bool = query.requires_grad  # type: ignore[attr-defined]
                score = query.call_method(
                    tx,
                    "new_empty",
                    [
                        VariableTracker.build(tx, []),
                    ],
                    {"requires_grad": VariableTracker.build(tx, scores_require_grad)},
                )
                new_args = [score, *bhmn]
            else:
                if fn_name != "mask_fn":
                    raise AssertionError("Illegal function name: " + fn_name)
                new_args = [*bhmn]

        with TransformGetItemToIndex():
            (
                (_body_output, _body_spec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn,
                new_args,
                {},  # expect only args no kwargs for now
                description=f"{self._HOP_NAME}: {fn_name}",
                source_target=self.value,
                set_subgraph_inputs="flatten_manual",
            )

        body_name = tx.output.install_subgraph(
            fn_name,
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)

        # It is possible that the score-mod function captures some free variables that are not
        # passed in as arguments. In this case, we need to lift them, which is handled by speculate_subgraph.
        # We then need to create proxies for this + the inputs.

        lifted_args = tuple(arg for arg in body_lifted_freevars)

        proxy_args = (body_node, lifted_args)

        return proxy_args

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Normalizes args to (query, key, value, score_mod, block_mask, scale,
        # kernel_options), traces score_mod and the mask function (no-op mask if none) into
        # subgraphs, proxies the tensor args, and emits the flex_attention call with subgraph
        # nodes and lifted buffers; the original aten op is set as context. Returns the wrapped
        # result (no example value).
        from .builder import wrap_fx_proxy

        (
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
        ) = self.normalize_to_args(list(args), kwargs)

        score_mod_node, score_mod_lifted_args = self.create_wrapped_node(
            tx, query, score_mod, "score_mod"
        )
        mask_fn = block_mask.items[-1]  # type: ignore[attr-defined]
        if mask_fn.is_python_constant() and mask_fn.as_python_constant() is None:
            mask_fn = VariableTracker.build(
                tx,
                torch.nn.attention.flex_attention.noop_mask,
                source=mask_fn.source,
            )
        mask_fn_node, mask_fn_lifted_args = self.create_wrapped_node(
            tx, query, mask_fn, "mask_fn"
        )

        proxied_args = [
            query,
            key,
            value,
            TupleVariable(block_mask.items[:-1], source=block_mask.source),  # type: ignore[attr-defined]
            scale,
            kernel_options,
        ]

        # Store the invocation as a call
        # Norm_kwargs contains the score_function and we don't want to proxy this because
        # Proxying user defined functions is not supported.
        inp_args, _ = proxy_args_kwargs(proxied_args, {})

        # Compose the ordered HOO args:
        # - inp_args: [query, key, value, block_mask, scale, kernel_options]
        # - subgraph node: [score_mod, mask_fn_node]
        # - lifted args from tracing subgraph: [score_mod_other_buffers, mask_fn_other_buffers]
        _, _, _, inp_arg_block_mask, inp_arg_scale, inp_arg_kernel_options = inp_args
        block_mask = tuple(inp_arg_block_mask + (mask_fn_node,))
        with torch.fx.experimental.proxy_tensor.set_original_aten_op(self.value):
            proxy = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    args=inp_args[:3]
                    + (
                        score_mod_node,
                        block_mask,
                        inp_arg_scale,
                        inp_arg_kernel_options,
                        score_mod_lifted_args,
                        mask_fn_lifted_args,
                    ),
                    kwargs={},
                ),
                example_value=None,
            )
        return proxy


@add_hop_context
class AutogradFunctionApplyVariable(VariableTracker):
    # [CLASS DOC] Variable representing an autograd.Function apply call: holds the forward and
    # backward callables and parent source. Calling it traces forward and backward into subgraphs,
    # aligns their signatures (inputs/outputs/saved tensors) and emits an autograd function apply
    # HOP call. Decorated with HOP context so errors carry the HOP name; falls back to eager on
    # graph breaks.
    _HOP_NAME: str = "autograd.Function"
    _ALLOW_FALLBACK_TO_EAGER = True

    def __init__(
        self, fwd_fn: Any, bwd_fn: Any, parent_source: Source | None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] Stores the forward and backward functions and parent source.
        super().__init__(**kwargs)
        self.fwd_fn = fwd_fn
        self.bwd_fn = bwd_fn
        self.parent_source = parent_source

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        # [FUNCTION DOC] Rich comparison using default object semantics (identity-based).
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the type as the builtin method type.
        return types.BuiltinMethodType

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces the autograd.Function apply: creates a forward tracer and context
        # variable, traces forward and backward subgraphs, rewires backward outputs to forward
        # inputs and saved tensors to backward inputs, records non-differentiable output indices,
        # dirty (mark_dirty) output indices (graph breaks if a dirty input is not returned), and
        # save_for_backward indices, installs both bodies as subgraphs, computes the example value
        # under fake mode, emits the apply call, reproxifies forward outputs and restores
        # requires_grad. Returns the forward output variable.
        """
        At the highest level, the goal of tracing an autograd.Function is to
        essentially emit a new autograd.Function object. To do this, Dynamo
        traces fwd and bwd graph and then inserts a AutogradFunctionApply HOP in
        the graph that call the traced fwd and bwd graph in the `forward` and
        `backward` methods respectively. AOTDispatcher desugars this HOP and
        just inlines the hop fwd and bwd into the main graph during its tracing.

        However, the traced forward and backward graphs cannot be directly
        placed in the new autograd.Function because autograd.Function has some
        requirements.

        a) # fwd graph inputs = # bwd graph outputs
        b) # fwd graph outputs = # bwd graph inputs
        c) Since the graphs do not have ctx variable, we have to manually return
        the saved_tensors from the forward and have additional inputs in the
        backward, and wire the connections.

        Unfortunately, reworking the initial traced fwd and bwd graphs to
        satisfy the above 3 conditions leads to a very tedious codebase.

        Lets look at an example

        class Foo:
            def __init__(self):
                self.a = 4

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, foo):
                ctx.save_for_backward(x)
                return x.sin() + foo.a

            @staticmethod
            def backward(ctx, grad):
                x, = ctx.saved_tensors
                return grad * x.cos()

        We want the resulting graphs to look like:

        # Note that Dynamo lifts the foo_a directly as an input.
        def fwd(ctx, x, foo_a):
            # (output, saved tensors / attrs)
            return (x.sin() + foo_a, (x))

        # Note that backward graph has None as the second output to match the
        # fwd requirements (even though the original backward function has just
        # output)
        def bwd(ctx, grad, x):
            return grad * x.cos(), None


        To accomplish this, we're going to:
        1. Construct a ctx object
        2. Speculate subgraph forward
        3. Speculate subgraph backward
        4. rewired_bwd_graph_inputs - Use the traced fwd graph as the anchor point, and rewire the backward graph outputs
        5. handle_saved_tensors_wiring - Handle the saved tensors, as mentioned in (c)
        """

        fwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
            tx.output,
            parent=tx.output.current_tracer,
            source_target=self._HOP_NAME,
        )

        ctx = self.prepare_ctx_vt(tx, args, kwargs)

        fwd_fn, fwd_out, fwd_graph, fwd_freevars, fwd_graph_output_vts = (
            self.trace_forward_graph(tx, ctx, fwd_tracer, args, kwargs)
        )

        bwd_args, bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts = (
            self.trace_backward_graph(tx, ctx, fwd_tracer, fwd_out, fwd_fn)
        )

        self.rewire_bwd_graph_outputs(
            fwd_freevars, bwd_out, bwd_graph, bwd_freevars, args
        )

        fwd_graph, bwd_graph = self.handle_saved_tensors_wiring(
            fwd_out,
            fwd_graph,
            fwd_freevars,
            fwd_graph_output_vts,  # type: ignore[arg-type]
            bwd_graph,
            bwd_freevars,
        )

        # If users call ctx.mark_non_differentiable, we should capture these output tensors who
        # are marked as non-differentiable and pass them to ApplyTemplate
        # at torch._functorch.autograd_function.AutogradFunctionApply for reconstruction.
        non_differentiable_idx = []
        if ctx.non_differentiable is not None:
            non_differentiable_set = set(ctx.non_differentiable)
            if not isinstance(fwd_out, variables.BaseListVariable):
                raise AssertionError("Expected fwd_out to be a BaseListVariable")
            for i, x in enumerate(fwd_out.items):
                if x.is_tensor() and x.as_proxy() in non_differentiable_set:
                    non_differentiable_idx.append(i)

        dirty_idx: list[int] = []
        if ctx.dirty_tensors is not None:
            dirty_tensor_set: set[Proxy] = {
                x.as_proxy() for x in ctx.dirty_tensors if x.is_tensor()
            }
            matched_dirty_tensor_set: set[Proxy] = set()
            if fwd_out.is_tensor():
                fwd_out_proxy = fwd_out.as_proxy()
                if fwd_out_proxy in dirty_tensor_set:
                    dirty_idx.append(0)
                    matched_dirty_tensor_set.add(fwd_out_proxy)
            else:
                if not isinstance(fwd_out, variables.BaseListVariable):
                    raise AssertionError("Expected fwd_out to be a BaseListVariable")
                for i, x in enumerate(fwd_out.items):
                    if x.is_tensor():
                        x_proxy = x.as_proxy()
                        if x_proxy in dirty_tensor_set:
                            dirty_idx.append(i)
                            matched_dirty_tensor_set.add(x_proxy)
            if dirty_tensor_set - matched_dirty_tensor_set:
                unimplemented(
                    gb_type="autograd.Function.apply: mark_dirty input not returned",
                    context="",
                    explanation="Dynamo only supports ctx.mark_dirty when each "
                    "dirty input is also returned from forward.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

        # See Note [Activations with no version counter checks in eager]
        # Compute which tensors in bwd_freevars came from ctx.save_for_backward.
        # This allows AOT autograd to distinguish between tensors saved via
        # save_for_backward vs those stashed directly on ctx (e.g., ctx.x = x).
        saved_for_backward_idx = []
        if ctx.saved_tensors is not None and len(ctx.saved_tensors.tensors) > 0:
            # Build a set of proxies that were passed to save_for_backward
            saved_tensor_proxies = OrderedSet()
            for tensor_vt in ctx.saved_tensors.tensors:
                if tensor_vt.is_tensor():
                    saved_tensor_proxies.add(tensor_vt.as_proxy())

            # bwd_freevars is a dict of outer-graph proxy -> inner-graph proxy
            # for all tensors passed from fwd to bwd. Find which indices
            # correspond to save_for_backward tensors.
            for i, fwd_proxy in enumerate(bwd_freevars.keys()):
                if fwd_proxy in saved_tensor_proxies:
                    saved_for_backward_idx.append(i)

        # Store fwd_body
        fwd_nn_modules = tx.output.tracing_context.module_context.copy_graphstate()
        fwd_name = tx.output.install_subgraph(
            "fwd_body",
            torch.fx.GraphModule(fwd_nn_modules.nn_modules, fwd_graph),
        )
        fwd_node = make_attr(tx, fwd_name)

        # Store bwd_body
        bwd_nn_modules = tx.output.tracing_context.module_context.copy_graphstate()
        bwd_name = tx.output.install_subgraph(
            "bwd_body",
            torch.fx.GraphModule(bwd_nn_modules.nn_modules, bwd_graph),
        )
        bwd_node = make_attr(tx, bwd_name)

        p_args = (
            fwd_node,
            bwd_node,
            *list(fwd_freevars.keys()),
        )
        kwargs_for_fn = {
            "non_differentiable_idx": non_differentiable_idx,
            "saved_for_backward_idx": saved_for_backward_idx,
        }
        # Preserve the existing HOP call shape for the common no-mark_dirty case.
        if dirty_idx:
            kwargs_for_fn["dirty_idx"] = dirty_idx

        # Store the invocation as a call
        from torch._functorch.autograd_function import autograd_function_apply

        # We use speculate_subgraph to get the fwd graph, but it's always under no grad mode like what eager mode does.
        # The fwd outputs (tensor's example_value) need to be inferred from fake tensor prop to get the correct attributes
        # (e.g, tensor.requires_grad), which would be used by downstream Dynamo tracing.
        # Since there can be other ops like Triton kernels, which depends on python dispatcher, we have to enable it.
        # TODO - revisit if we need the python dispatcher
        with enable_python_dispatcher():
            with tx.output.fake_mode:
                fwd_freevars_args = [_get_fake_value(arg) for arg in fwd_freevars]

                example_value = autograd_function_apply(
                    tx.output.nn_modules[fwd_node.node.name],
                    tx.output.nn_modules[bwd_node.node.name],
                    *fwd_freevars_args,
                    **kwargs_for_fn,
                )

        flat_variable = add_call_function(
            tx, autograd_function_apply, p_args, kwargs_for_fn, example_value
        )
        # type: ignore[arg-type]
        overwrite_tensor_vt_proxy(fwd_graph_output_vts, flat_variable)
        # type: ignore[arg-type]
        overwrite_tensor_vt_requires_grad(fwd_graph_output_vts, flat_variable)
        return fwd_out

    def prepare_ctx_vt(
        self, tx: "InstructionTranslatorBase", args: Any, kwargs: Any
    ) -> "AutogradFunctionContextVariable":
        # [FUNCTION DOC] Creates the autograd function context variable and, under discarded graph
        # changes, gives it a dummy proxy (with the context value as example value) so it can be
        # passed to subgraph speculation. Returns the context variable.
        from . import AutogradFunctionContextVariable

        ctx = AutogradFunctionContextVariable.create(tx, args, kwargs)
        with discard_graph_changes(tx):
            # A little hacky, but we need a dummy ctx proxy for speculate_subgraph.
            # We should clean this up at some point.
            proxy = tx.output.create_proxy(
                "call_function", torch.autograd.function.FunctionCtx, (), {}
            )
            # type: ignore[attr-defined]
            set_example_value(proxy.node, ctx.value)
            # type: ignore[attr-defined]
            ctx.proxy = proxy
        # pyrefly: ignore[bad-return]
        return ctx

    def trace_forward_graph(
        self,
        tx: "InstructionTranslatorBase",
        ctx: "AutogradFunctionContextVariable",
        fwd_tracer: "SubgraphTracer",
        args: Any,
        kwargs: Any,
    ) -> tuple[
        VariableTracker,
        VariableTracker,
        torch.fx.Graph,
        dict[Proxy, Proxy],
        VariableTracker | tuple[VariableTracker, ...],
    ]:
        # [FUNCTION DOC] Wraps forward in a trace helper (no-grad and view handling) and
        # speculates it with automatic output flattening, allowing side effects and filtering
        # aliased intermediates, using the forward tracer. Lifts unused tensor args as inputs so
        # their gradients are accounted for, and graph breaks if the context's non-differentiable
        # grad materialization flag was mutated. Returns (forward fn variable, output variable,
        # graph, lifted free variables, graph output variables).
        """
        Traces the forward method of the autograd.Function object.
        """
        from torch._functorch.autograd_function import DynamoAutogradFunctionTraceHelper

        fwd_fn, fwd_args = self.prepare_fn_vt(tx, ctx, "forward", args)

        # autograd.Function forward does a few things like running in no_grad
        # mode and also applying view_as for input tensors that are returned as
        # outputs. Therefore, we wrap the original forward in a helper that have
        # those extra bits for Dynamo to trace.
        fwd_fn = _make_inlined(tx, DynamoAutogradFunctionTraceHelper.fwd_trace_helper)(
            fwd_fn
        )

        # Speculate subgraph on the fwd
        fwd_out, fwd_graph, fwd_freevars, fwd_graph_output_vts, _ = (
            speculate_subgraph_with_auto_output_flattening(
                tx,
                fwd_fn,
                fwd_args,
                kwargs,
                self._HOP_NAME,
                enable_grad=None,
                set_subgraph_inputs="automatic",
                allow_side_effects=True,
                filter_aliased_intermediates=True,
                tracer=fwd_tracer,
            )
        )

        # There could be unused inputs in the forward, and Dynamo might not
        # capture them. We must lift them as inputs, because even though they
        # are not used in forward, we still need to account for their gradients
        # in the backward.
        for arg in args:
            if arg.is_tensor():
                fwd_tracer.maybe_lift_tracked_freevar_to_input(arg.as_proxy())

        if ctx in tx.output.side_effects.store_attr_mutations:
            if (
                "_materialize_non_diff_grads"
                in tx.output.side_effects.store_attr_mutations[ctx]
            ):
                unimplemented(
                    gb_type="autograd.Function.apply: _materialize_non_diff_grads mutation",
                    context="",
                    explanation="Mutations to autograd.Function.ctx._materialize_non_diff_grads are not supported.",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

        return fwd_fn, fwd_out, fwd_graph, fwd_freevars, fwd_graph_output_vts

    def trace_backward_graph(
        self,
        tx: "InstructionTranslatorBase",
        ctx: "AutogradFunctionContextVariable",
        fwd_tracer: "SubgraphTracer",
        fwd_out: VariableTracker,
        fwd_fn: VariableTracker,
    ) -> tuple[
        list[VariableTracker],
        VariableTracker,
        torch.fx.Graph,
        dict[Proxy, Proxy],
        VariableTracker | tuple[VariableTracker, ...],
    ]:
        # [FUNCTION DOC] Speculates backward with a tracer that is a child of the forward tracer,
        # using forced tensor inputs (one per forward output, none placeholders for non-tensor
        # outputs), grad disabled and no side effects; side effects are restored afterwards. If
        # tracing hits unknown properties in the backward, retries with a rewritten backward and
        # relaxed banned-op config (graph breaks if backward is not a function or method). Returns
        # (backward args, output variable, graph, lifted free variables, graph output variables).
        """
        Traces the backward method of the autograd.Function object.
        """
        from . import UserMethodVariable

        # Note that for the forward, we do not restore side effects, because we
        # want the later tracing to see the side-effects. But for backward, we
        # are just trying to capture the graph, and therefore we must restore
        # the side effects.
        prev_side_effects = tx.output.side_effects

        # Speculate subgraph on the backward. We make the bwd tracer a child of
        # the fwd tracer, because backward may rely on tensors/attrs created in
        # the fwd tracer.
        bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
            tx.output,
            parent=fwd_tracer,
            source_target=self._HOP_NAME,
        )

        bwd_args = []
        if fwd_out.is_tensor():
            bwd_args.append(fwd_out)
        else:
            if not isinstance(fwd_out, variables.BaseListVariable):
                raise AssertionError("Expected fwd_out to be a BaseListVariable")
            for i in fwd_out.items:
                if i.is_tensor():
                    bwd_args.append(i)
                else:
                    bwd_args.append(ConstantVariable.create(None))

        bwd_fn, bwd_args = self.prepare_fn_vt(tx, ctx, "backward", bwd_args)

        def is_strict_for(v: VariableTracker) -> bool:
            # [FUNCTION DOC] HELPER: Strictness predicate for translation mode: tensors created in
            # the forward tracer are lax (false), everything else strict (true).
            if v.is_tensor():
                # we can be more lax for stuff from forward
                return v.proxy.tracer is not fwd_tracer  # type: ignore[attr-defined]
            return True

        # automatic_with_forced_inputs relies on the function arg names to
        # create a new proxy. Also, it will always INSERT a tensor placeholder
        # as input, even though it might not be used in the graph. This allows
        # us to make a mapping for the backward graph.
        with (
            tx.output.subtracer(fwd_fn, fwd_tracer),  # type: ignore[arg-type]
            tx.strict_translation_mode(is_strict_for),
        ):
            try:
                bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts, _ = (
                    speculate_subgraph_with_auto_output_flattening(
                        tx,
                        bwd_fn,
                        bwd_args,
                        {},
                        self._HOP_NAME,
                        # TODO - revisit if we need enable_grad
                        enable_grad=False,
                        set_subgraph_inputs="automatic_with_forced_inputs",
                        allow_side_effects=False,
                        tracer=bwd_tracer,
                    )
                )
            except torch._dynamo.exc.UnknownPropertiesDuringBackwardTrace as e:
                # TODO - Do not support this path because of eager
                # divergence forced by contiguous calls. Instead suggested
                # nonstrict_trace.
                from unittest import mock

                bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
                    tx.output,
                    parent=fwd_tracer,
                    source_target=self._HOP_NAME,
                )
                from .._trace_wrapped_higher_order_op import (
                    autograd_function_backward_rewritten,
                )
                from .builder import SourcelessBuilder

                if isinstance(self.bwd_fn, types.FunctionType):
                    bwd_fn = SourcelessBuilder.create(
                        tx, autograd_function_backward_rewritten(self.bwd_fn)
                    )
                elif isinstance(self.bwd_fn, types.MethodType):
                    bwd_fn = UserMethodVariable(
                        autograd_function_backward_rewritten(self.bwd_fn.__func__),
                        VariableTracker.build(tx, self.bwd_fn.__class__),
                    )
                else:
                    unimplemented(
                        gb_type="autograd.Function.apply: non-function or method backward (2)",
                        context=str(self.bwd_fn),
                        explanation="Expected backward function to be a function or method.",
                        hints=[],
                        from_exc=e,
                    )

                with mock.patch(
                    "torch._dynamo.config._autograd_backward_strict_mode_conditional_banned_ops",
                    [],
                ):
                    bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts, _ = (
                        speculate_subgraph_with_auto_output_flattening(
                            tx,
                            bwd_fn,
                            bwd_args,
                            {},
                            self._HOP_NAME,
                            enable_grad=False,
                            set_subgraph_inputs="automatic_with_forced_inputs",
                            allow_side_effects=False,
                            tracer=bwd_tracer,
                        )
                    )

        # Restore the side effects
        tx.output.side_effects = prev_side_effects

        return bwd_args, bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts

    def rewire_bwd_graph_outputs(
        self,
        fwd_freevars: dict[Proxy, Proxy],
        bwd_out: VariableTracker,
        bwd_graph: torch.fx.Graph,
        bwd_freevars: dict[Proxy, Proxy],
        orig_fwd_args: list[VariableTracker],
    ) -> None:
        # [FUNCTION DOC] Rewrites the backward graph's output node in place so its outputs line up
        # one-to-one with the forward graph inputs (in lifted free variable order): each forward
        # tensor arg maps to its backward output node (or none), other positions get none. Graph
        # breaks if the same fake tensor is passed to several forward inputs. Lints the graph.
        # ---------------------------------------------------------------------
        # Forward–Backward Input/Output Alignment
        #
        # autograd.Function requires that the outputs of backward() correspond
        # exactly to the inputs of forward(). Normally this alignment is the
        # user’s responsibility. However, when Dynamo synthesizes a new
        # autograd.Function for a traced region, Dynamo must perform this
        # alignment automatically.
        #
        # To do this, Dynamo uses the *original* forward call site as the anchor
        # that defines how forward inputs map to backward outputs.
        #
        # ---------------------------------------------------------------------
        # Terminology
        #
        # fwd_freevars / bwd_freevars:
        #     Maps from *outer-graph proxies* to *inner-graph placeholder
        #     proxies*. Keys are always outer-graph proxies (these may be actual
        #     user inputs or intermediate values lifted into the subgraph).
        #
        # orig_fwd_args:
        #     VariableTrackers for the forward() inputs. Since these correspond
        #     to user-exposed arguments, each tracker points to an *outer-graph*
        #     proxy.
        #
        # bwd_outs:
        #     VariableTrackers for the backward() outputs. These usually point to
        #     *inner-graph* proxies, except for cases where a forward input is
        #     passed directly through to a backward output—in which case the
        #     tracker may still refer to an outer-graph proxy.
        #
        # ---------------------------------------------------------------------
        # Goal
        #
        # To ensure forward–backward consistency, we must rewire the backward
        # graph outputs so that they line up with the forward graph inputs.
        #
        # We build a mapping from outer-graph proxy → inner-graph proxy using
        # orig_fwd_args and bwd_outs, then iterate over the fwd_graph inputs to
        # determine which backward outputs must be generated (or padded with
        # None) to satisfy autograd’s calling convention.
        #
        # ---------------------------------------------------------------------
        # Example
        #
        # Suppose the forward receives a user-defined object:
        #
        # @dataclass
        # class Weird:
        #     x: int
        #     b: torch.Tensor
        #     c: torch.Tensor
        #
        # class Foo(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, x: torch.Tensor, weird: Weird, z: torch.Tensor):
        #         ctx.save_for_backward(weird.b, weird.c)
        #         return weird.b * weird.c * x.clone()
        #
        #     @staticmethod
        #     def backward(ctx, grad):
        #         b, c = ctx.saved_tensors
        #         return grad * b * c, None, grad * 2
        #
        # Dynamo lifts the tensor fields of the user-defined object for the trace:
        #
        # fwd_graph():
        #     %l_weird_b : FakeTensor = placeholder[target=l_weird_b]
        #     %l_weird_c : FakeTensor = placeholder[target=l_weird_c]
        #     %l_x_      : FakeTensor = placeholder[target=l_x_]
        #     %l_z_      : FakeTensor = placeholder[target=l_z_]
        #     ...
        #     return (outs,)
        #
        # The initial backward graph:
        #
        # bwd_graph():
        #     %grad       : Tensor    = placeholder[target=grad]
        #     %l_weird_b  : FakeTensor = placeholder[target=l_weird_b]
        #     %l_weird_c  : FakeTensor = placeholder[target=l_weird_c]
        #     ...
        #     return (mul_1, mul_2)
        #
        # The forward graph has 4 inputs, but the backward graph produces only 2
        # outputs, and their ordering does not match the forward argument order.
        #
        # So Dynamo rewires the backward graph outputs to align with the forward
        # inputs:
        #
        # bwd_graph():
        #     ...
        #     return (None, None, mul_1, mul_2)
        #
        # This ensures the synthesized autograd.Function conforms to PyTorch’s
        # forward/backward contract.
        # ---------------------------------------------------------------------

        def get_bwd_node(vt: VariableTracker) -> torch.fx.Node:
            # [FUNCTION DOC] HELPER: Returns the backward graph node for a tensor variable,
            # translating through the lifted free variable mapping when it is a backward input.
            # Backward tensor vt here can be - (1) an intermediate, or (2) input
            # to the backward graph. If it is an input to the backward graph, we have to lookup bwd_freevars to get the inner proxy.
            return bwd_freevars.get(vt.proxy, vt.proxy).node  # type: ignore[attr-defined]

        tensor_args_seen: list[Any] = []
        for arg in orig_fwd_args:
            if arg.is_tensor():
                fake_tensor = _get_fake_value(arg)
                if any(fake_tensor is seen for seen in tensor_args_seen):
                    unimplemented(
                        gb_type="autograd.Function.apply: duplicate tensor input",
                        context="",
                        explanation="Dynamo does not support tracing autograd.Function.apply "
                        "when the same tensor is passed to multiple forward inputs.",
                        hints=[
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )
                tensor_args_seen.append(fake_tensor)

        # Find the mapping between orig_fwd_args and bwd_out
        # pyrefly: ignore [implicit-any]
        outer_fwd_proxy_to_bwd_node = {}
        if isinstance(bwd_out, variables.BaseListVariable):
            bwd_outs = bwd_out.items
            for idx, fwd_arg in enumerate(orig_fwd_args):
                # We care about tensor args. For non-tensor args, the bwd output returns None.
                if fwd_arg.is_tensor():
                    bwd_out_at_idx = bwd_outs[idx]
                    if bwd_out_at_idx.is_tensor():
                        # type: ignore[attr-defined]
                        outer_fwd_proxy_to_bwd_node[fwd_arg.proxy] = get_bwd_node(
                            bwd_out_at_idx
                        )
                    else:
                        # backward can return None at the output
                        if not isinstance(bwd_out_at_idx, variables.ConstantVariable):
                            raise AssertionError(
                                "Expected backward output to be a ConstantVariable"
                            )
                        if bwd_out_at_idx.value is not None:
                            raise AssertionError(
                                "Expected backward output ConstantVariable to have value None"
                            )
                        # type: ignore[attr-defined]
                        outer_fwd_proxy_to_bwd_node[fwd_arg.proxy] = None

        elif bwd_out.is_tensor():
            # type: ignore[attr-defined]
            outer_fwd_proxy_to_bwd_node[orig_fwd_args[0].proxy] = get_bwd_node(bwd_out)

        # Ideally, we should have walked through the fwd placeholders. But we
        # can instead walk through the fwd_freevars, which is a insertion sorted
        # dictionary and therefore represents the outer_proxies for the
        # placeholder in the same order as that as placeholders.
        rewired_bwd_outputs = [
            outer_fwd_proxy_to_bwd_node.get(fwd_proxy) for fwd_proxy in fwd_freevars
        ]

        for node in bwd_graph.find_nodes(op="output"):
            bwd_graph.erase_node(node)
            break
        bwd_graph.output(tuple(rewired_bwd_outputs))
        bwd_graph.lint()

    def handle_saved_tensors_wiring(
        self,
        fwd_out: VariableTracker,
        fwd_graph: torch.fx.Graph,
        fwd_freevars: dict[Proxy, Proxy],
        fwd_graph_body_outputs: list[VariableTracker],
        bwd_graph: torch.fx.Graph,
        bwd_freevars: dict[Proxy, Proxy],
    ) -> tuple[torch.fx.Graph, torch.fx.Graph]:
        # [FUNCTION DOC] Aligns forward outputs with backward inputs and wires saved tensors:
        # maps forward tensor outputs to backward placeholders (none for tensors inside objects,
        # later turned into unused placeholders), appends saved/lifted tensors from the backward
        # free variables, changes the forward output to (existing outputs, extra saved outputs),
        # and builds a new backward graph with the rewired placeholder list followed by the
        # copied body. Mutates the forward graph; returns (forward graph, new backward graph).
        # Raises AssertionError if a backward placeholder is missing from the environment.
        # ---------------------------------------------------------------------
        # Rewiring Forward Outputs to Backward Inputs (and Handling Saved Tensors)
        #
        # In `rewire_bwd_graph_outputs`, we aligned the *forward inputs* with the
        # *backward outputs*. This method performs the complementary task:
        # aligning the *forward outputs* with the *backward inputs*, while also
        # incorporating all tensors saved via ctx.save_for_backward.
        #
        # There are two main issues we must resolve:
        #
        # (1) Forward outputs may contain non-tensor values.
        #     This means the number of tensors visible in fwd_out may not match
        #     the number of tensors produced by the traced forward graph. As a
        #     result, the backward graph’s placeholders may not line up with the
        #     actual tensor outputs.
        #
        # (2) The backward graph may require intermediate tensors saved during
        #     the forward pass (via save_for_backward), but those intermediates
        #     might not currently be included among the forward graph’s outputs.
        #
        # Together, these issues mean that the bwd_graph input signature may be
        # inconsistent with what fwd_graph outputs, and we need to rewrite both.
        #
        # Lets look at an example to understand the transformation
        #
        # class Add(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, x, y):
        #         a = torch.sin(x)
        #         b = torch.cos(y)
        #         ctx.save_for_backward(a)
        #         return Foo(a, b), x * y

        #     @staticmethod
        #     def backward(ctx, grad_a, grad_b):
        #         (a,) = ctx.saved_tensors
        #         return grad_b * 2, a * grad_b * 3

        # Before
        # fwd_graph():
        #     %l_x_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_x_]
        #     %l_y_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_y_]
        #     ....
        #     return (a, b, out)
        #
        # bwd_graph():
        #     %grad_b : torch.Tensor [num_users=2] = placeholder[target=grad_b]
        #     %a : torch._subclasses.fake_tensor.FakeTensor [num_users=1] = placeholder[target=a]
        #     ....
        #     return (mul, mul_2)
        #
        # The problems here:
        #   (1) fwd_graph has 3 tensor outputs (a, b, out), but bwd_graph has
        #       only 1 gradient input - grad_b. We need 3.
        #
        #   (2) bwd_graph uses `a` (a saved tensor) as an input, but fwd_graph
        #       does not currently return `a`. To make `a` available to the
        #       backward graph, the forward graph must expose it as part of its
        #       output signature.
        #
        # After this transformation
        # fwd_graph():
        #     %l_x_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_x_]
        #     %l_y_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_y_]
        #     .....
        #     return ((a, b, out), (a,))
        # bwd_graph():
        #     %unused_0 : [num_users=0] = placeholder[target=unused_0]
        #     %unused_1 : [num_users=0] = placeholder[target=unused_1]
        #     %grad_b : [num_users=2] = placeholder[target=grad_b]
        #     %a : [num_users=1] = placeholder[target=a]
        #     .....
        #     return (mul, mul_2)
        #
        # Key changes:
        #
        #   1) The forward graph now returns:
        #           (existing_outputs), (saved_tensors)
        #      This exposes saved intermediates (`a`) as part of the fwd output
        #      structure, making them available to backward.
        #
        #   2) The backward graph input signature is rewritten to:
        #           (*grads_for_existing_outputs, *saved_tensors)
        #      This ensures the counts and ordering match the new fwd_graph
        #      output structure. Placeholders corresponding to tensors whose
        #      gradients are unused (e.g., `a`, `b`) appear as `%unused_*`.
        #
        # This alignment ensures that the synthesized autograd.Function follows
        # PyTorch’s forward/backward calling convention and that all required
        # saved tensors are available to the backward graph.
        # ---------------------------------------------------------------------

        # To address Problem (1), we must determine which backward-graph inputs
        # correspond to the forward-graph outputs.
        #
        # We use two facts:
        #   • `fwd_out` preserves the original forward output order.
        #   • Backward-graph inputs are also ordered according to the backward()
        #     method signature, thanks to automatic_with_forced_inputs.
        #
        # For any forward output that is *not* a tensor, there is no
        # corresponding tensor placeholder in the backward graph. During tracing,
        # we intentionally inserted a `None` VariableTracker for these positions,
        # so the backward graph contains no placeholder for them.
        bwd_input_nodes = list(bwd_graph.find_nodes(op="placeholder"))
        # pyrefly: ignore [implicit-any]
        fwd_vt_to_bwd_node = {}
        bwd_idx = 0
        if isinstance(fwd_out, variables.BaseListVariable):
            for fwd_vt in fwd_out.items:
                if fwd_vt.is_tensor():
                    fwd_vt_to_bwd_node[fwd_vt] = bwd_input_nodes[bwd_idx]
                    bwd_idx += 1
        else:
            if fwd_out.is_tensor():
                fwd_vt_to_bwd_node[fwd_out] = bwd_input_nodes[bwd_idx]
                bwd_idx += 1

        rewired_bwd_graph_inputs = []
        for fwd_graph_vt in fwd_graph_body_outputs:
            # for tensor vts that were part of a user-defined object (like in
            # the above example), we just set None for now. Later, we will use
            # these None to insert a unused placeholder.
            # type: ignore[arg-type]
            rewired_bwd_graph_inputs.append(fwd_vt_to_bwd_node.get(fwd_graph_vt))

        # To address Problem (2), we must incorporate any tensors that were saved
        # (or otherwise smuggled) from the forward pass into the backward graph.
        #
        # Fortunately, these are easy to identify: they appear in `bwd_freevars`.
        # `bwd_freevars` maps outer-graph lifted proxies to inner-graph placeholder
        # proxies. Because the backward graph is traced using proxies originating
        # from `fwd_out`, any value lifted into the backward graph represents a
        # saved/smuggled tensor.
        #
        # Once we identify these saved tensors, we must also locate their
        # corresponding forward-graph proxies so that the forward graph can return
        # these tensors as part of its output signature.
        extra_fwd_output_nodes = []
        for fwd_proxy, bwd_inner_proxy in bwd_freevars.items():
            # For backward, its easy, just get the node from bwd_inner_proxy
            rewired_bwd_graph_inputs.append(bwd_inner_proxy.node)

            # For the fwd_proxy, it could be a proxy from the outer graph, or it
            # could be an intermediate.
            # First ensure that's its inner fwd proxy
            inner_fwd_proxy = fwd_freevars.get(fwd_proxy, fwd_proxy)
            extra_fwd_output_nodes.append(inner_fwd_proxy.node)

        # Mechanical steps from here on. We have the extra_fwd_outputs and rewired_bwd_inputs. Lets make the changes.
        # Lets change the fwd graph outputs.
        # pyrefly: ignore [implicit-any]
        fwd_output_nodes = []
        for node in fwd_graph.find_nodes(op="output"):
            fwd_output_nodes = node.args[0]
            fwd_graph.erase_node(node)
            break

        # The signature is now ((*existing_outputs), (*extra_outputs)). Please
        # take a look at AutogradFunctionApply where we take the saved_tensors
        # out in the forward method to save for backward.
        new_fwd_graph_outputs = (fwd_output_nodes, tuple(extra_fwd_output_nodes))
        fwd_graph.output(new_fwd_graph_outputs)
        fwd_graph.lint()

        # Now lets change the bwd graph.
        new_graph = torch.fx.Graph()
        env = {}

        count = itertools.count()

        for node in rewired_bwd_graph_inputs:
            if node is None:
                new_node = new_graph.placeholder(f"unused_{next(count)}")
            else:
                new_node = new_graph.placeholder(node.name)
                new_node.meta = copy.copy(node.meta)
            env[node] = new_node

        for node in bwd_graph.nodes:
            if node.op == "placeholder":
                if node not in env:
                    raise AssertionError("Placeholder node not found in env")
            else:
                env[node] = new_graph.node_copy(node, lambda x: env[x])
                env[node].meta = copy.copy(node.meta)

        new_graph.lint()
        return fwd_graph, new_graph

    def prepare_fn_vt(
        self,
        tx: "InstructionTranslatorBase",
        ctx: "AutogradFunctionContextVariable",
        method_name: str,
        args: list[VariableTracker],
    ) -> tuple[VariableTracker, list[VariableTracker]]:
        # [FUNCTION DOC] Builds a callable variable for the forward or backward function (plain
        # function, or bound method represented via a user method variable on its class) with a
        # source derived from the parent source, and the argument list with the class (for methods)
        # and context prepended. Graph breaks if the function is neither a function nor a method.
        # Returns (function variable, argument list).
        from . import UserMethodVariable

        source = None
        if self.parent_source:
            source = AttrSource(self.parent_source, member=method_name)

        if method_name == "forward":
            fn = self.fwd_fn
        else:
            fn = self.bwd_fn

        fn_vt, fn_args = None, None
        if isinstance(fn, types.FunctionType):
            fn_vt = VariableTracker.build(tx, fn, source=source)
            fn_args = [ctx, *args]
        elif isinstance(fn, types.MethodType):
            cls_vt = VariableTracker.build(tx, fn.__class__)
            fn_vt = UserMethodVariable(
                fn.__func__,
                cls_vt,
                source=source,
            )
            fn_args = [cls_vt, ctx, *args]
        else:
            unimplemented(
                gb_type="autograd.Function.apply: non-function or method forward",
                context=str(fn),
                explanation=f"Expected {method_name} to be a function or method.",
                hints=[],
            )
        if fn_vt is None:
            raise AssertionError("fn_vt must not be None")
        if fn_args is None:
            raise AssertionError("fn_args must not be None")
        return fn_vt, fn_args


def _get_fake_value(x: Union[VariableTracker, Proxy, "FakeTensor"]) -> "FakeTensor":
    # [FUNCTION DOC] HELPER: Returns the example (fake) value of a variable or proxy; any other
    # input is returned unchanged.
    if isinstance(x, variables.VariableTracker):
        return x.as_proxy().node.meta["example_value"]
    elif isinstance(x, torch.fx.Proxy):
        return x.node.meta["example_value"]
    else:
        return x


def maybe_positional_arg_names(func: VariableTracker) -> list[str] | None:
    # [FUNCTION DOC] Returns the names of the positional parameters of the underlying python
    # function of a callable variable (a parameter named self is renamed with a trailing
    # underscore), or none if the variable has no function, the signature is unavailable, or it has
    # variadic positionals.
    result = []
    if not hasattr(func, "get_function"):
        return None
    try:
        fn = func.get_function()
    except (Unsupported, NotImplementedError):
        return None
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return None
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if (
            param.kind is inspect.Parameter.POSITIONAL_ONLY
            or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            if name == "self":
                # FX graphs can't have a placeholder named self
                result.append("self_")
            else:
                result.append(name)
    return result


class BaseHOPVariable(WrapHigherOrderVariable):
    # [CLASS DOC] Generic variable for BaseHOP operators without a dedicated class: traces the
    # first argument as a wrapped body and passes keyword args as proxies. The HOP name is taken
    # from the operator at construction. Mutation and aliasing unsupported.
    # Generic fallback for BaseHOP instances not explicitly mapped
    # The actual HOP name comes from self.value._name at runtime
    _HOP_NAME = "base HOP (name not yet determined)"
    supports_input_mutation = False
    supports_aliasing = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Initializes the parent and sets the instance HOP name from the operator's
        # name.
        super().__init__(*args, **kwargs)
        self._HOP_NAME = self.value._name

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the type of the wrapped operator.
        return type(self.value)

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Traces args[0] with the remaining positional args as a wrapped body
        # (subgraph named subgraph), asserts no leftover kwargs from wrapping, proxies the call's
        # keyword args and emits the HOP call with auto output flattening. Returns the output
        # variable.
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            _,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx, args[0], args[1:], {}, self.value._name, subgraph_name="subgraph"
        )
        if len(p_kwargs) != 0:
            raise AssertionError("Expected p_kwargs to be empty")

        p_kwargs = {key: value.as_proxy() for key, value in kwargs.items()}
        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
        )


class LocalMapWrappedHigherOrderVariable(WrapHigherOrderVariable):
    # [CLASS DOC] Variable for the local_map HOP wrapper used with distributed tensors. Wrapping is
    # gated by a class-level enabled flag toggled with the enable context manager. Mutation and
    # aliasing unsupported.
    _HOP_NAME = "torch.ops.higher_order.local_map_hop"
    supports_input_mutation = False
    supports_aliasing = False

    # Subclasses aren't supported by speculate_subgraph yet
    # So this HOP is only usable with plain tensors
    _enabled = False

    @classmethod
    @contextlib.contextmanager
    def enable(cls) -> Generator[None, None, None]:
        # [FUNCTION DOC] Class-level context manager that sets the enabled flag for the block and
        # restores the previous value on exit (also on exceptions).
        # KNOWN ISSUE: the docstring usage example refers to enable_wrapping, but the method is
        # named enable.
        """Context manager to temporarily enable local map wrapping.
        Will be removed when speculate_subgraph supports subclass inputs:
        https://github.com/pytorch/pytorch/issues/161456.

        Usage:
            with LocalMapWrappedHigherOrderVariable.enable_wrapping():
                # Code where should_wrap_in_hop will return True
                pass
        """
        old_value = cls._enabled
        cls._enabled = True
        try:
            yield
        finally:
            cls._enabled = old_value

    @classmethod
    def should_wrap_in_hop(cls, value: Any) -> bool:
        # [FUNCTION DOC] Returns true only if distributed is available, the value has the type of
        # the local-map wrapper function, is that exact function, and wrapping is enabled.
        if not torch.distributed.is_available():
            return False

        from torch.distributed.tensor.experimental._func_map import _local_map_wrapped

        # check is important to avoid subclass dispatch
        if type(value) is not type(_local_map_wrapped):
            return False

        return value is _local_map_wrapped and cls._enabled

    @staticmethod
    # pyrefly: ignore[bad-override]
    def build(**options: Any) -> "TorchHigherOrderOperatorVariable":
        # [FUNCTION DOC] Static factory that creates the variable for the local_map HOP through the
        # generic make factory, forwarding the given options (such as source).
        return TorchHigherOrderOperatorVariable.make(
            torch._higher_order_ops.local_map_hop,
            **options,
        )

    def python_type(self) -> type:
        # [FUNCTION DOC] Reports the type of the wrapped operator.
        return type(self.value)

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Rewrites a local_map call into the local_map HOP. Unpacks (user function,
        # out/in/in-grad placements, device mesh, redistribute flag, spmd flag, user args); requires
        # trailing-only none placements, a provided device mesh and matching input count (raises
        # AssertionError otherwise). Temporarily replaces input example values with local-shaped
        # tensors, traces the function with forced inputs, validates input/output counts and
        # argument ordering against placements, emits the HOP call, restores global-shaped input
        # values, converts output example values back to global shapes, and stores the placement
        # information in the body graph module metadata. Returns the output variable.
        """
        Goal of this function is to rewrite local_map usage as a HOP:
            local_map(func, ...) -> local_map_hop(gm, ...)
        """

        (
            user_func,
            out_placements,
            in_placements,
            in_grad_placements,
            device_mesh,
            redistribute_inputs,
            enable_spmd_types,
            *user_args,
        ) = args

        # None placements are used to pass non-Tensors into the local_map function.
        # Containers passed this way can not hold tensors. Thus, Dynamo would have inlined
        # into them, and we handle None placements by assuming they will be desugared away.
        # This will need to be adjusted for dynamic shapes support.
        def check_none_last(placements: Sequence[Any | None]) -> int:
            # [FUNCTION DOC] HELPER: Counts none placements and raises AssertionError if a non-none
            # placement follows a none one.
            seen_none = 0
            for p in placements:
                if p is None:
                    seen_none += 1
                else:
                    if seen_none != 0:
                        raise AssertionError(
                            "Tracing local_map is only currently supported with None placements last."
                        )
            return seen_none

        inputs_none_placements = check_none_last(in_placements.value)  # type: ignore[attr-defined]
        output_none_placements = check_none_last(out_placements.value)  # type: ignore[attr-defined]

        local_map_kwargs = {
            "out_placements": out_placements.value,  # type: ignore[attr-defined]
            "in_placements": in_placements.value,  # type: ignore[attr-defined]
            "redistribute_inputs": redistribute_inputs.value,  # type: ignore[attr-defined]
            "in_grad_placements": in_grad_placements.value,  # type: ignore[attr-defined]
            "device_mesh": device_mesh.value,  # type: ignore[attr-defined]
        }
        if local_map_kwargs["device_mesh"] is None:
            raise AssertionError(
                "Not yet implemented, please manually provide a device_mesh to local_map."
            )
        mesh = local_map_kwargs["device_mesh"]

        # For Autoparallel, the initial trace is done with global shapes, then we decide model weights sharding,
        # and reuse the graph. Since the sharding decision is after the initial trace, we can't trace with local shapes.
        # For local_map however, since we specify all placements, we can trace with local shapes.

        # Step 1: Validate the annotated function matches the input_placements (i.e. that it can run in eager)
        template = (
            "Expecting {expected} {inputs_or_outputs} to local_map function based on placements"
            ", but found {actual}. Please ensure the count matches for eager. "
        )
        if len(in_placements.value) != len(user_args):  # type: ignore[attr-defined]
            raise AssertionError(
                template.format(
                    expected=len(in_placements.value),  # type: ignore[attr-defined]
                    inputs_or_outputs="inputs",
                    actual=len(user_args),
                )
            )

        from torch._higher_order_ops.local_map import (
            redistribute_fw_inputs,
            redistribute_fw_outputs,
        )

        # Step 2: Convert inputs to local shapes
        priors = {}
        for placements, vt in zip(in_placements.value, user_args):  # type: ignore[attr-defined]
            if isinstance(vt, variables.lazy.LazyVariableTracker):
                vt = variables.lazy.LazyVariableTracker.realize_all(vt)

            if not vt.is_tensor():
                if placements is not None:
                    raise AssertionError(
                        "Expected placements to be None for non-tensor input"
                    )
                continue

            global_tensor = vt.as_proxy().node.meta["example_value"]
            # NOTE: We don't support local_map region relying on exact grad_fn information
            # This is okay since accessing grad_fn is a graph break.
            local_tensor = redistribute_fw_inputs(
                (global_tensor,),
                (placements,),
                mesh,
            )
            local_tensor = local_tensor[0]

            priors[vt] = global_tensor
            vt.as_proxy().node.meta["example_value"] = local_tensor
            # pyrefly: ignore [missing-attribute]
            vt.synchronize_attributes(tx)

        # Step 3: Trace local_map subgraph with local tensors
        #
        # Use "automatic_with_forced_inputs" so subgraph placeholders are
        # created eagerly in the user's call-site arg order. Under "automatic",
        # Dynamo lifts freevars in first-use order, which diverges between
        # static and dynamic shapes (e.g. `x.shape[-1]` early in the body lifts
        # `x` first under dynamic shapes but constant-folds under static). The
        # forced mode keeps `in_placements` aligned with user args regardless.
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_gmod,
            body_name,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            user_func,
            user_args,
            kwargs,
            self.value._name,
            subgraph_name="subgraph",
            set_subgraph_inputs="automatic_with_forced_inputs",
        )

        # Step 4: Validate traced graph signature still matches placement information
        expected_num_inputs = len(in_placements.value) - inputs_none_placements  # type: ignore[attr-defined]
        actual_num_inputs = len(body_gmod.graph.find_nodes(op="placeholder"))
        expected_num_outputs = len(out_placements.value) - output_none_placements  # type: ignore[attr-defined]
        if len(body_gmod.graph.find_nodes(op="output")) != 1:
            raise AssertionError("Expected exactly one output node in body graph")
        actual_num_outputs = len(body_gmod.graph.find_nodes(op="output")[0].args[0])

        template = (
            "Expecting {expected} {inputs_or_outputs} to local_map function based on placements"
            ", but found {actual}. If the count matches for eager, "
            "Dynamo may have flattened {inputs_or_outputs} to the function or found additional "
            "tensors used via closures. "
            "Please adjust the input placements to match what the traced graph sees: \n{gm_str}."
        )

        def make_error_msg(*args: Any) -> str:
            # [FUNCTION DOC] HELPER: Formats the mismatch error message from (expected count,
            # actual count, "inputs"/"outputs") including the readable body graph.
            expected_num, actual_num, inputs_or_outputs = args
            gm_str = body_gmod.print_readable(print_output=False)
            return template.format(
                expected=expected_num,
                inputs_or_outputs=inputs_or_outputs,
                actual=actual_num,
                gm_str=gm_str,
            )

        if expected_num_inputs != actual_num_inputs:
            raise AssertionError(
                make_error_msg(expected_num_inputs, actual_num_inputs, "inputs")
            )
        if expected_num_outputs != actual_num_outputs:
            raise AssertionError(
                make_error_msg(expected_num_outputs, actual_num_outputs, "outputs")
            )

        if inputs_none_placements > 0:
            expected_input_nodes = [
                arg.as_proxy().node for arg in user_args[:-inputs_none_placements]
            ]
        else:
            expected_input_nodes = [arg.as_proxy().node for arg in user_args]
        actual_input_nodes = [proxy.node for proxy in p_args]
        if actual_input_nodes[0].op != "get_attr":
            raise AssertionError("Expected first actual input node op to be 'get_attr'")
        if "subgraph" not in actual_input_nodes[0].target:  # type: ignore[attr-defined]
            raise AssertionError(
                "Expected 'subgraph' in first actual input node target"
            )
        if expected_input_nodes != actual_input_nodes[1:]:
            raise AssertionError(
                "local_map: unexpected argument ordering. Outer HOP args did "
                "not match user call-site order "
                f"(expected={expected_input_nodes}, actual={actual_input_nodes[1:]}). "
                "automatic_with_forced_inputs should have preserved order; "
                "please file an issue at https://github.com/pytorch/pytorch/issues."
            )
        if len(p_kwargs) != 0:
            raise AssertionError("Expected p_kwargs to be empty")

        # Step 5: Install local_map subgraph
        p_kwargs = {key: value.as_proxy() for key, value in kwargs.items()}
        out = _call_function_with_auto_output_flattening(
            tx,
            self.value,
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
        )

        # Step 6: Restore inputs and outputs to global shapes
        for vt, global_tensor in priors.items():
            vt.as_proxy().node.meta["example_value"] = global_tensor
            # pyrefly: ignore [missing-attribute]
            vt.synchronize_attributes(tx)

        outs = out.items if isinstance(out, TupleVariable) else [out]
        if len(outs) != len(out_placements.value):  # type: ignore[attr-defined]
            raise AssertionError(
                "Number of outputs must match number of out_placements"
            )
        for placements, vt in zip(out_placements.value, outs):  # type: ignore[attr-defined]
            if not vt.is_tensor():  # type: ignore[attr-defined]
                if placements is not None:
                    raise AssertionError(
                        "Expected placements to be None for non-tensor output"
                    )
                continue

            local_tensor = vt.as_proxy().node.meta["example_value"]  # type: ignore[attr-defined]

            # NOTE: We don't support code after the local_map region relying on exact grad_fn information
            # This is okay since accessing grad_fn is a graph break.
            global_tensor = redistribute_fw_outputs(
                (local_tensor,),
                (placements,),
                mesh,
                num_activations=0,  # this is not the joint
            )
            global_tensor = global_tensor[0]

            vt.as_proxy().node.meta["example_value"] = global_tensor  # type: ignore[attr-defined]
            vt.synchronize_attributes(tx)  # type: ignore[attr-defined]

        # TODO: Figure out how to handle output order diverging from eager

        # Treat as const, so we don't have to deal with Placement types in fx IR
        # Guarded with EQUALS_MATCH on local_map call's arguments
        body_gmod.meta["local_map_kwargs"] = {
            "out_placements": out_placements.value[:expected_num_outputs],  # type: ignore[attr-defined]
            "in_placements": in_placements.value[:expected_num_inputs],  # type: ignore[attr-defined]
            "redistribute_inputs": redistribute_inputs.value,  # type: ignore[attr-defined]
            "in_grad_placements": in_grad_placements.value,  # type: ignore[attr-defined]
            "device_mesh": device_mesh.value,  # type: ignore[attr-defined]
        }
        if out is None:
            raise AssertionError("out must not be None")
        return out


from .invoke_subgraph import InvokeSubgraphHigherOrderVariable


# Map operator names to their corresponding variable for fast TorchHigherOrderOperatorVariable.make()
_hop_name_to_variable_class = {
    "cond": CondHigherOrderVariable,
    "switch": SwitchHigherOrderVariable,
    "while_loop": WhileLoopHigherOrderVariable,
    "while_loop_stack_output": WhileLoopStackOutputHigherOrderVariable,
    "map_impl": MapHigherOrderVariable,
    "executorch_call_delegate": ExecutorchCallDelegateHigherOrderVariable,
    "out_dtype": OutDtypeHigherOrderVariable,
    "wrap": WrapHigherOrderVariable,
    "hints_wrapper": HintsWrapperHigherOrderVariable,
    "flex_gemm": FlexGemmHigherOrderVariable,
    "flex_attention": FlexAttentionHigherOrderVariable,
    "flex_attention_backward": FlexAttentionBackwardHighOrderVariable,
    "wrap_activation_checkpoint": CheckpointHigherOrderVariable,
    "tag_activation_checkpoint": CheckpointHigherOrderVariable,
    "_export_tracepoint": ExportTracepointHigherOrderVariable,
    "trace_wrapped": TraceWrappedHigherOrderOperatorVariable,
    "strict_mode": StrictModeHigherOrderVariable,
    "run_with_rng_state": RunWithRNGStateHigherOrderVariable,
    "associative_scan": AssociativeScanHigherOrderVariable,
    "scan": ScanHigherOrderVariable,
    "call_torchbind": CallTorchbindHigherOrderVariable,
    "print": PrintHigherOrderVariable,
    "wrap_with_set_grad_enabled": WrapWithSetGradEnabledHigherOrderVariable,
    "wrap_with_autocast": WrapWithAutocastHigherOrderVariable,
    "dynamo_bypassing_wrapper": DynamoBypassingWrapperHigherOrderVariable,
    "auto_functionalized": AutoFunctionalizeHigherOrderVariable,
    "auto_functionalized_v2": AutoFunctionalizeHigherOrderVariable,
    "inline_asm_elementwise": InlineAsmElementwiseHigherOrderVariable,
    "invoke_subgraph": InvokeSubgraphHigherOrderVariable,
    "custom_function_call": CustomFunctionHigherOrderOperatorVariable,
    "local_map_hop": LocalMapWrappedHigherOrderVariable,
}
