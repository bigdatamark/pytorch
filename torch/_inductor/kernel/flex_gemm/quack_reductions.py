# mypy: allow-untyped-defs
# [MODULE DOC] Helpers that lower FlexGEMM grouped local reductions into QuACK/CuTeDSL
# epilogue text. Defines the grouped TensorSSA layout (M or N groups), the physical
# reduction callback record, FX-to-CuTeDSL argument and call translation,
# shape/pointwise predicates, and per-node lowering functions (view/reshape, grouped
# split/select, squeeze, getitem, online softmax preparation, reductions). Main caller
# is the emitter in fx_cutedsl_codegen. No import-time side effects beyond definitions.
"""Lower FlexGEMM grouped local reductions into QuACK/CuTeDSL epilogues.

FlexGEMM recognizes a narrow local-reduction contract inside the GEMM output
tile: an epilogue reshapes the accumulator to expose contiguous groups along M
or N, then reduces only that grouped dimension. N-axis groups up to one 32-lane
fragment lower as ordinary in-fragment TensorSSA reductions; larger N groups
produce TensorSSA partials that QuACK combines physically.
M-axis groups currently always use QuACK's physical row-lane/warp combine path,
even when the group is small enough to fit in one fragment. Inductor owns FX
normalization and output contracts; these helpers describe the supported
TensorSSA shapes and generated combine/finalize expressions QuACK needs.

The main caller is ``materialize_flex_gemm_epilogue`` in ``fx_cutedsl_codegen.py``.
Shared GEMM epilogue analysis recognizes grouped layouts and reductions;
materialization wraps accepted geometries in ``GroupedTensorSSALayout`` and
routes FX nodes through ``lower_view_or_reshape``,
``lower_prepare_softmax_online``, and ``lower_tensorssa_reduce``.
"""

import dataclasses
import math
from typing import Any, cast

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLOpOverrides,
    tensorssa_reduction,
)
from torch._inductor.kernel.flex_gemm.constraints import (
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionGeometry,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import normalize_shape
from torch._inductor.ops_handler import ReductionType
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout(GemmReductionGeometry):
    # [CLASS DOC] Frozen grouped M/N TensorSSA layout (extends GemmReductionGeometry,
    # which supplies group, axis, group_size and the physical-callback predicates). Its
    # methods produce CuTeDSL shape/size expression text for reshaping a fragment into
    # groups and for keepdim results, and the reduction profile for the axis.
    """Describe a grouped M/N TensorSSA view inside the generated epilogue."""

    def fragment_group_size_expr(self, source: Any) -> str:
        # [FUNCTION DOC] Return expression text for the group size usable in the current
        # fragment: the configured group size capped by the first-mode size of source.
        """Return the local group size available in this epilogue fragment."""
        return (
            f"cutlass.const_expr(min({self.group_size}, "
            f"cute.size({source}.shape, mode=[0])))"
        )

    def fragment_repeat_expr(self, source: Any) -> str:
        # [FUNCTION DOC] Return expression text for how many repeats of the capped group
        # cover the fragment's first mode of source.
        """Return the repeat count needed to cover the current epilogue fragment."""
        return (
            f"cutlass.const_expr(cute.size({source}.shape, mode=[0]) "
            f"// min({self.group_size}, cute.size({source}.shape, mode=[0])))"
        )

    def tensorssa_shape(self, source: Any) -> str:
        # [FUNCTION DOC] Return the grouped TensorSSA shape text for source: group
        # dimension in the second nested slot for axis 1 (N groups), first slot
        # otherwise (M groups), with the repeat count in the third slot.
        fragment_group_size = self.fragment_group_size_expr(source)
        repeats = self.fragment_repeat_expr(source)
        if self.axis == 1:
            return f"((1, {fragment_group_size}, {repeats}), 1, 1)"
        return f"(({fragment_group_size}, 1, {repeats}), 1, 1)"

    def keepdim_shape(self, source: Any) -> str:
        # [FUNCTION DOC] Return the keepdim shape text (group dimension collapsed to 1,
        # repeats kept) for source.
        return f"((1, 1, {self.fragment_repeat_expr(source)}), 1, 1)"

    @property
    def needs_physical_combine(self) -> bool:
        # [FUNCTION DOC] Property: alias of needs_physical_callbacks, true when the
        # reduction must be finished by QuACK physical combine/finalize callbacks.
        return self.needs_physical_callbacks

    @property
    def reduction_profile(self) -> str:
        # [FUNCTION DOC] Property: reduction profile text marking which nested mode is
        # reduced, the second slot for axis 1 and the first slot otherwise.
        if self.axis == 1:
            return "((None, 1, None), 1, 1)"
        return "((1, None, None), 1, 1)"


def _cute_op_name(target: Any) -> str | None:
    # [FUNCTION DOC] HELPER: Derive the op name for an FX call target: overload packet
    # name, string as given, or callable's name, else None. The name div is mapped to
    # truediv.
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


@dataclasses.dataclass(frozen=True)
class FlexGemmPhysicalReduction:
    # [CLASS DOC] Frozen description of a QuACK physical reduction: combine_expr is the
    # generated combine expression for two partial values and finalize_expr (default:
    # identity on the value) is applied to the combined result.
    """Describe QuACK's physical combine/finalize callback for local reductions."""

    combine_expr: str
    finalize_expr: str = "value"


FLEX_GEMM_POINTWISE_OP_NAMES = frozenset(
    (
        "_to_copy",
        "clamp",
        "clamp_max",
        "clamp_min",
        "convert_element_type",
        "inline_asm_elementwise",
    )
)


def _cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    # [FUNCTION DOC] HELPER: Convert an FX argument to a CuTeDSL value. Nodes are looked
    # up in env (NotImplementedError if absent); non-finite floats become source text
    # for nan or +/-inf; ints, floats, bools, strings and torch
    # dtype/device/layout/memory_format pass through; tuples and lists convert
    # recursively keeping their type; anything else raises NotImplementedError.
    """Translate FX node references and constants into CuTeDSL epilogue values."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'float("nan")'
        return 'float("inf")' if value > 0 else 'float("-inf")'
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            str,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _generate_like(
    kernel: Any, expr: str, ref: Any, shape_ref: Any | None = None
) -> Any:
    # [FUNCTION DOC] HELPER: Emit expr into the kernel body through its CSE with dtype
    # taken from ref and shape from shape_ref (default ref). Returns the new CSE
    # variable.
    """Emit CuTeDSL while preserving dtype and shape metadata from references."""
    if shape_ref is None:
        shape_ref = ref
    return kernel.cse.generate(
        kernel.body,
        expr,
        dtype=getattr(ref, "dtype", None),
        shape=getattr(shape_ref, "shape", None),
    )


def _keepdim_and_broadcast(
    kernel: Any, reduced: Any, layout: GroupedTensorSSALayout, source: Any
) -> tuple[Any, Any]:
    # [FUNCTION DOC] HELPER: Emit two values for a grouped reduction result: the reduced
    # value reshaped to the keepdim shape, and that value broadcast to the shape of
    # source. Returns (keepdim value, broadcast value). Writes lines to the kernel body.
    """Materialize keepdim and store-shaped forms of a grouped reduction."""
    keepdim_source = _generate_like(
        kernel, f"{reduced}.reshape({layout.keepdim_shape(source)})", reduced
    )
    return keepdim_source, _generate_like(
        kernel, f"{keepdim_source}.broadcast_to({source}.shape)", keepdim_source, source
    )


def _cute_call(
    node: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    # [FUNCTION DOC] HELPER: Lower one FX call by invoking the same-named method on the
    # active ops handler. For inline asm elementwise it renames the asm_str keyword, and
    # adds input dtypes and per-input scalar flags derived from node tensor metadata
    # (NotImplementedError if an input lacks tensor metadata). Raises
    # NotImplementedError for unknown op names or missing handler methods. Returns the
    # handler result.
    """Lower one FX call through the active CuTeDSL operations handler."""
    target = node.target
    op_name = _cute_op_name(target)
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    if op_name == "inline_asm_elementwise":
        # The HOP spells the asm text `asm_str`; the ops handler spells it `asm`.
        kwargs = dict(kwargs)
        kwargs["asm"] = kwargs.pop("asm_str")
        input_values = []
        for input_node in node.args[: len(args)]:
            value = (
                input_node.meta.get("val")
                if isinstance(input_node, torch.fx.Node)
                else None
            )
            if not isinstance(value, torch.Tensor):
                raise NotImplementedError(
                    "FlexGEMM inline asm inputs require tensor metadata"
                )
            input_values.append(value)
        kwargs["input_dtypes"] = tuple(value.dtype for value in input_values)
        kwargs["scalar_sources"] = tuple(
            all(isinstance(dim, int) and dim == 1 for dim in value.shape)
            for value in input_values
        )
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def _local_reduce_store_arg(
    value: Any, env: dict[torch.fx.Node, Any], sources: dict[torch.fx.Node, Any]
) -> Any:
    # [FUNCTION DOC] HELPER: Convert a value for use in a compressed-store pointwise
    # expression: nodes with a store source use it, tuples and lists recurse, everything
    # else uses the normal argument conversion.
    if isinstance(value, torch.fx.Node) and value in sources:
        return sources[value]
    if isinstance(value, (tuple, list)):
        return type(value)(
            _local_reduce_store_arg(item, env, sources) for item in value
        )
    return _cute_arg(value, env)


def tensor_meta_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    # [FUNCTION DOC] Return the node's tensor shape from fake-value metadata as a tuple,
    # or None if the value is not a tensor.
    """Return fake-tensor shape metadata when the FX value is tensor-like."""
    meta = node.meta.get("val")
    if isinstance(meta, torch.Tensor):
        return tuple(meta.shape)
    return None


def node_preserves_tensor_shapes(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] Return True if a pointwise node keeps the shape of at least one
    # tensor input equal to its output and every other tensor input broadcasts to
    # exactly the output shape (shapes compared as strings). False when the output has
    # no tensor shape or a broadcast fails or widens an input.
    """Reject pointwise broadcasts that cannot preserve a grouped TensorSSA input."""
    output_shape = tensor_meta_shape(node)
    if output_shape is None:
        return False
    output_shape_key = tuple(str(dim) for dim in output_shape)
    has_same_shape_input = False
    for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
        input_shape = tensor_meta_shape(input_node)
        if input_shape is None:
            continue
        input_shape_key = tuple(str(dim) for dim in input_shape)
        if input_shape_key == output_shape_key:
            has_same_shape_input = True
            continue
        try:
            broadcast_shape = get_broadcasted_shape(input_shape_key, output_shape_key)
        except AssertionError:
            return False
        if broadcast_shape is None or tuple(broadcast_shape) != output_shape_key:
            return False
    return has_same_shape_input


def is_pointwise_node(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] Return True for call_function nodes that are tagged pointwise or
    # whose op name is in the FlexGEMM extra pointwise set.
    if node.op != "call_function":
        return False
    return (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    ) or _cute_op_name(node.target) in FLEX_GEMM_POINTWISE_OP_NAMES


def is_shape_preserving_pointwise_node(node: torch.fx.Node) -> bool:
    # [FUNCTION DOC] Return True if the node is pointwise and preserves its tensor
    # shapes.
    return is_pointwise_node(node) and node_preserves_tensor_shapes(node)


def lower_view_or_reshape(
    node: torch.fx.Node,
    normalized: NormalizedView,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    active_grouped_layouts: OrderedSet[GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    preserve_value_layout: bool = False,
) -> Any | None:
    # [FUNCTION DOC] Lower an analyzed view or reshape. If the source has a store
    # source, propagate it to node and forward the source value. If node has a grouped
    # layout that is active and not preserve_value_layout, emit a reshape to the grouped
    # shape; if inactive or preserved, forward the source unchanged. If only the source
    # is grouped, forward the source. Otherwise return None (not handled).
    """Emit an analyzed view using grouped provenance."""
    source_node = normalized.source
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    source = _cute_arg(source_node, env)
    grouped_layout = grouped_tensors.get(node)
    if grouped_layout is not None:
        if preserve_value_layout or grouped_layout not in active_grouped_layouts:
            return source
        return _generate_like(
            kernel,
            f"{source}.reshape({grouped_layout.tensorssa_shape(source)})",
            source,
        )
    if source_node in grouped_tensors:
        return source
    return None


def lower_grouped_n_split(
    node: torch.fx.Node,
    normalized: NormalizedSplit,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
) -> tuple[Any, ...] | None:
    # [FUNCTION DOC] Lower a split whose node has an axis-1 grouped layout: reshape the
    # source to the grouped shape and return a tuple of per-group slices (one per group
    # index). Returns None if the node has no layout or the axis is not 1.
    """Split an analyzed output contraction into grouped TensorSSA values."""
    layout = grouped_tensors.get(node)
    if layout is None or layout.axis != 1:
        return None
    source = _cute_arg(normalized.source, env)
    grouped = _generate_like(
        kernel, f"{source}.reshape({layout.tensorssa_shape(source)})", source
    )
    return tuple(
        _generate_like(
            kernel,
            f"{grouped}[((0, {index}, None), None, None)]",
            grouped,
        )
        for index in range(layout.group)
    )


def lower_grouped_n_select(
    normalized: NormalizedSelect,
    index: int,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
) -> Any:
    # [FUNCTION DOC] Emit one slice of the source selecting the given group index in the
    # grouped axis-1 layout and return the generated value. Assumes analysis already
    # validated the index.
    """Select one analysis-validated output-contraction group value."""
    source = _cute_arg(normalized.source, env)
    return _generate_like(
        kernel,
        f"{source}[((0, {index}, None), None, None)]",
        source,
    )


def lower_full_scalar(node: torch.fx.Node) -> Any | None:
    # [FUNCTION DOC] Return the constant of an aten full call that builds a scalar
    # (empty shape) from a bool, int or float fill value; return None in every other
    # case.
    if node.op != "call_function" or node.target is not torch.ops.aten.full.default:
        return None
    shape = normalize_shape(node.args[0])
    if shape != ():
        return None
    value = node.args[1]
    return value if isinstance(value, (bool, int, float)) else None


def lower_squeeze(
    node: torch.fx.Node,
    normalized: NormalizedSqueeze,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    # [FUNCTION DOC] Forward a squeeze as an alias of its source value, copying any
    # store source to node. Returns None if the source has not been lowered into env.
    """Forward an analyzed squeeze alias."""
    source_node = normalized.source
    if source_node not in env:
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
    return _cute_arg(source_node, env)


def lower_getitem(
    node: torch.fx.Node,
    normalized: NormalizedGetItem,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    # [FUNCTION DOC] Lower getitem on a lowered tuple or list value: return the element
    # (also propagating the indexed store source if the source has one). Returns None if
    # the source is not a sequence or the index is out of range.
    """Index an analyzed aggregate value."""
    source_node, index = normalized.source, normalized.index
    source = _cute_arg(source_node, env)
    if not isinstance(source, (tuple, list)) or not -len(source) <= index < len(source):
        return None
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node][
            index
        ]
    return source[index]


def lower_prepare_softmax_online(
    node: torch.fx.Node,
    normalized: NormalizedPrepareSoftmax,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any:
    # [FUNCTION DOC] Lower online softmax preparation over a grouped tensor: emit a max
    # reduction, a broadcast of it, the centered values, their exponential, a sum
    # reduction and its broadcast. Records and returns the pair (max, sum) of broadcast
    # values as the node's store sources. Raises NotImplementedError if the input is not
    # grouped, needs physical callbacks, or the reduced dim is not the grouped dim.
    """Lower analyzed online-softmax preparation."""
    input_node, dim = normalized.source, normalized.dim
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if layout.needs_physical_callbacks:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: prepare_softmax_online "
            "needs a multi-value generated physical reducer"
        )
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: prepare_softmax_online "
            "currently supports only the grouped dimension"
        )
    source = _cute_arg(input_node, env)
    max_reduced = _generate_like(
        kernel,
        f'{source}.reduce(cute.ReductionOp.MAX, init_val=float("-inf"), reduction_profile={layout.reduction_profile})',
        source,
    )
    _, max_store = _keepdim_and_broadcast(kernel, max_reduced, layout, source)
    centered = _generate_like(kernel, f"({source} - {max_store})", source)
    exp_centered = CuteDSLOpOverrides.exp(centered)
    sum_reduced = _generate_like(
        kernel,
        f"{exp_centered}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={layout.reduction_profile})",
        exp_centered,
    )
    _, sum_store = _keepdim_and_broadcast(kernel, sum_reduced, layout, source)
    local_reduce_store_sources[node] = (max_store, sum_store)
    return max_store, sum_store


def lower_tensorssa_reduce(
    node: torch.fx.Node,
    normalized: NormalizedReduction,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSALayout],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction],
) -> Any:
    # [FUNCTION DOC] Lower an analyzed sum/max/min/mean style reduction over a grouped
    # tensor. Raises NotImplementedError for an explicit dtype, an ungrouped input, or a
    # dim that is not the grouped one. When physical callbacks are needed, records a
    # physical reduction (mean finalizes by dividing by the group size); for axis 0 (M
    # groups) it returns the unreduced source and records it as the store source.
    # Otherwise emits an in-fragment reduce (mean divides by the group size), records
    # the broadcast store source and returns the keepdim form if keepdim else the
    # reduced value.
    """Lower an analyzed reduction while deferring physical finalization."""
    input_node = normalized.source
    dim = normalized.dim
    keepdim = normalized.keepdim
    dtype = normalized.dtype
    reduction_type = normalized.reduction_type
    if dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    if input_node not in grouped_tensors:
        raise NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)
    layout = grouped_tensors[input_node]
    if not layout.matches_reduction_dim(dim):
        raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
    reduction_name = cast(
        ReductionType, "sum" if reduction_type == "mean" else reduction_type
    )
    desc = tensorssa_reduction(reduction_name)
    finalize_expr = f"value / {layout.group}.0" if reduction_type == "mean" else "value"
    source = _cute_arg(input_node, env)
    needs_physical_callbacks = layout.needs_physical_callbacks
    if needs_physical_callbacks:
        local_reduce_physical_reductions[node] = FlexGemmPhysicalReduction(
            desc.combine_expr, finalize_expr
        )
        if layout.axis == 0:
            local_reduce_store_sources[node] = source
            return source
    reduced = _generate_like(
        kernel,
        f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, reduction_profile={layout.reduction_profile})",
        source,
    )
    if reduction_type == "mean" and not needs_physical_callbacks:
        reduced = _generate_like(
            kernel, f"{reduced} / {float(layout.group)!r}", reduced
        )
    keepdim_source, local_reduce_store_sources[node] = _keepdim_and_broadcast(
        kernel, reduced, layout, source
    )
    if keepdim:
        return keepdim_source
    return reduced
