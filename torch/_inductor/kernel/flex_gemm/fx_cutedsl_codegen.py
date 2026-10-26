# mypy: allow-untyped-defs
# [MODULE DOC] Analysis and code generation for the FlexGEMM epilogue FX graph. Provides
# the analysis entry point (finds the single GEMM node, classifies main/aux/local-reduce
# outputs, plans output contractions over grouped N columns) and the emitter that lowers
# each FX node to CuTeDSL source text plus optional physical reduction combine/finalize
# callbacks. Import-time effects: only defines aliases to gemm_epilogue_analysis types;
# no registration.
"""Analyze FlexGEMM epilogue FX graphs and materialize CuTeDSL source.

``analyze_flex_gemm_epilogue`` indexes FX dependencies, identifies nodes that
carry grouped TensorSSA layouts, matches supported local reductions, and plans
the main, auxiliary, and local-reduction consumers.

``materialize_flex_gemm_epilogue`` uses that analysis to generate the CuTeDSL
epilogue and physical reduction callbacks.
"""

import dataclasses
import hashlib
from typing import Any

from sympy import Max, Min

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    upcast_compute_type,
    use_cutedsl_fast_math,
)
from torch._inductor.kernel import gemm_epilogue_analysis as _epilogue_analysis
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_CHUNKED_OUTPUT_CONTRACTION_REDUCE_ERROR,
    FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR,
    FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR,
    FLEX_GEMM_OUTPUT_TENSOR_ERROR,
    FlexGemmLocalReduceGeometry,
    FlexGemmOutputContraction,
    LOCAL_REDUCE_AUX_TENSORSSA_ERROR,
    LOCAL_REDUCE_COMBINE_FN_SUFFIX,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FINALIZE_FN_SUFFIX,
    LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR,
    LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR,
    local_reduce_unsupported_tensorssa_error,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _cute_op_name,
    _keepdim_and_broadcast,
    _local_reduce_store_arg,
    FlexGemmPhysicalReduction,
    GroupedTensorSSALayout,
    is_shape_preserving_pointwise_node,
    lower_full_scalar,
    lower_getitem,
    lower_grouped_n_select,
    lower_grouped_n_split,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmReductionPlan,
    iter_fx_node_inputs,
    NormalizedDtypeView,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_codegen import (
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    guarded_int,
    statically_known_shape_equal,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


FlexGemmEpilogueGraph = _epilogue_analysis.GemmEpilogueGraph
FlexGemmLocalReduceAnalysis = _epilogue_analysis.GemmLocalReduceAnalysis
FlexGemmLocalReduceMatch = _epilogue_analysis.GemmLocalReduceMatch
FlexGemmLocalReduceStore = _epilogue_analysis.GemmLocalReduceStore
FlexGemmOutputLocalReducePlan = _epilogue_analysis.GemmOutputLocalReducePlan


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan(_epilogue_analysis.GemmOutputPlan):
    # [CLASS DOC] Frozen output plan extending the base GemmOutputPlan with
    # output_contraction (grouped-column contraction descriptor or None) and
    # output_storage (the physical node under a terminal dtype view, or None).
    """Extend output planning with contraction and terminal storage metadata."""

    output_contraction: FlexGemmOutputContraction | None = None
    output_storage: torch.fx.Node | None = None


FlexGemmCuteDSLKernel = GemmEpilogueCuteDSLKernel


class FlexGemmCuteDSLOpOverrides(GemmEpilogueCuteDSLOpOverrides):
    # [CLASS DOC] CuTeDSL op overrides for FlexGEMM; differs from the base only in
    # clamp, clamp_min and clamp_max, which emit max/min calls that propagate NaN.
    """Add FlexGEMM-specific NaN-propagating clamp semantics."""

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        # [FUNCTION DOC] Clamp x to optional lower bound min and upper bound max by
        # chaining clamp_min then clamp_max; either bound may be None (skipped). Returns
        # the resulting CuTeDSL value; no side effects.
        result = x
        if min is not None:
            result = FlexGemmCuteDSLOpOverrides.clamp_min(result, min)
        if max is not None:
            result = FlexGemmCuteDSLOpOverrides.clamp_max(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        # [FUNCTION DOC] Elementwise maximum of x and min using the base binary-op
        # helper with a NaN-propagating math max template. Returns the generated value.
        return GemmEpilogueCuteDSLOpOverrides._apply_binary_op(
            x,
            min,
            "cutlass_math.max({a}, {b}, propagate_nan=True)",
            Max,
        )

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        # [FUNCTION DOC] Elementwise minimum of x and max using the base binary-op
        # helper with a NaN-propagating math min template. Returns the generated value.
        return GemmEpilogueCuteDSLOpOverrides._apply_binary_op(
            x,
            max,
            "cutlass_math.min({a}, {b}, propagate_nan=True)",
            Min,
        )


def tuple_output_plan(
    output: Any,
    aux_outputs: tuple[Any, ...],
    analysis: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    # [FUNCTION DOC] Build the output plan for an epilogue returning a main output plus
    # aux outputs. Raises NotImplementedError if any output is not an FX node, if more
    # than one aux output is a compressed local-reduce store, or if a feed-main
    # reduction and a compressed aux reduce different physical values. A compressed aux
    # match becomes the plan's local_reduce (merged with a common feed-main match, which
    # then feeds the main output) and is removed from the aux list; otherwise a feed-
    # main-only plan or a plain plan is returned.
    """Classify multi-output epilogues after checking local-reduce consumers."""
    if not isinstance(output, torch.fx.Node) or not all(
        isinstance(aux_output, torch.fx.Node) for aux_output in aux_outputs
    ):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_TENSOR_ERROR)
    feed_match = analysis.common_feed_main_match((output, *aux_outputs))
    compressed_aux_plans = tuple(
        (index, match, plan)
        for index, aux_output in enumerate(aux_outputs)
        if (match := analysis.matches.get(aux_output)) is not None
        if (plan := analysis.compressed_aux_plan(output, aux_output, index)) is not None
    )
    if len(compressed_aux_plans) > 1:
        raise NotImplementedError(LOCAL_REDUCE_MIXED_MATCH_ERROR)
    if compressed_aux_plans:
        local_reduce_index, compressed_match, compressed_aux_plan = (
            compressed_aux_plans[0]
        )
        if feed_match is not None:
            if feed_match.value_node is not compressed_match.value_node:
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            compressed_aux_plan = feed_match.to_plan(
                store=FlexGemmLocalReduceStore(
                    aux_outputs[local_reduce_index], local_reduce_index
                ),
                feeds_main=True,
            )
        return FlexGemmOutputPlan(
            output,
            tuple(
                aux_output
                for index, aux_output in enumerate(aux_outputs)
                if index != local_reduce_index
            ),
            local_reduce=compressed_aux_plan,
        )
    feed_main_plan = analysis.feed_main_output_plan(output, aux_outputs)
    if feed_main_plan is not None:
        return FlexGemmOutputPlan(
            feed_main_plan.output,
            feed_main_plan.aux_outputs,
            feed_main_plan.local_reduce,
        )
    return FlexGemmOutputPlan(output, aux_outputs)


def output_plan(
    graph_module: torch.fx.GraphModule,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> FlexGemmOutputPlan:
    # [FUNCTION DOC] Classify the graph's single output node into a FlexGemmOutputPlan
    # using the shared local-reduce analysis. A tuple of one element is unwrapped;
    # longer tuples are delegated to tuple_output_plan. Raises NotImplementedError if
    # there is not exactly one output node or the output is not a tensor node. Returns a
    # plain plan unless a feed-main reduction applies.
    """Classify output consumers from one shared local-reduce analysis."""
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            return tuple_output_plan(output, tuple(aux_outputs), local_reduce)
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    feed_main_plan = local_reduce.feed_main_output_plan(output_value)
    if feed_main_plan is None:
        return FlexGemmOutputPlan(output_value)
    return FlexGemmOutputPlan(
        feed_main_plan.output,
        feed_main_plan.aux_outputs,
        feed_main_plan.local_reduce,
    )


def bind_terminal_output_storage(
    graph: FlexGemmEpilogueGraph, outputs: FlexGemmOutputPlan
) -> FlexGemmOutputPlan:
    # [FUNCTION DOC] If the main output is a normalized dtype view, verify its source
    # has the same dtype size, shape and strides as the view result and record the
    # source as output_storage in a copy of the plan; otherwise return the plan
    # unchanged. Raises NotImplementedError when the view is not a pure same-width
    # reinterpretation (or metadata is missing).
    """Record the physical value beneath a terminal same-width dtype view."""
    normalized = graph.normalized_nodes.get(outputs.output)
    if not isinstance(normalized, NormalizedDtypeView):
        return outputs
    source_meta = normalized.source.meta.get("val")
    output_meta = outputs.output.meta.get("val")
    if (
        not isinstance(source_meta, torch.Tensor)
        or not isinstance(output_meta, torch.Tensor)
        or normalized.dtype is not output_meta.dtype
        or source_meta.dtype.itemsize != output_meta.dtype.itemsize
        or not statically_known_shape_equal(source_meta.shape, output_meta.shape)
        or not statically_known_shape_equal(source_meta.stride(), output_meta.stride())
    ):
        raise NotImplementedError(
            "FlexGEMM terminal dtype views must preserve shape, stride, and element size"
        )
    return dataclasses.replace(outputs, output_storage=normalized.source)


@dataclasses.dataclass(frozen=True)
class OutputContractionUse:
    # [CLASS DOC] Frozen record of one matched use of grouped GEMM columns in the main
    # output: source node, group count, whether access is chunked (split/select over
    # groups) rather than interleaved, the group indices consumed, and an optional node
    # to register as a grouped layout.
    """Describe one supported use of grouped GEMM values in the main output."""

    source: torch.fx.Node
    group: int
    chunked: bool
    group_indices: tuple[int, ...]
    layout_node: torch.fx.Node | None = None


@dataclasses.dataclass(frozen=True)
class OutputContractionPlan:
    # [CLASS DOC] Frozen staged output contraction: the contraction descriptor, the
    # mapping from select nodes to their group index, and grouped layouts to register.
    # Nothing is applied to the analysis until commit is called.
    """Stage one complete output contraction until analysis commits it."""

    contraction: FlexGemmOutputContraction
    select_indices: dict[torch.fx.Node, int]
    grouped_tensors: dict[torch.fx.Node, FlexGemmLocalReduceGeometry]

    def commit(
        self,
        outputs: FlexGemmOutputPlan,
        local_reduce: FlexGemmLocalReduceAnalysis,
    ) -> FlexGemmOutputPlan:
        # [FUNCTION DOC] Validate and install the staged contraction. Raises
        # NotImplementedError if the plan already has aux outputs or a local reduce, or
        # if a chunked contraction output depends on any matched reduction. Side effect:
        # merges the staged grouped layouts into local_reduce.grouped_tensors. Returns a
        # copy of outputs with output_contraction set.
        """Validate composition and install this output contraction."""
        if outputs.aux_outputs or outputs.local_reduce is not None:
            raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_COMPOSITION_ERROR)
        if self.contraction.chunked and any(
            local_reduce.graph.depends_on(outputs.output, reduced)
            for reduced in local_reduce.matches
        ):
            raise NotImplementedError(FLEX_GEMM_CHUNKED_OUTPUT_CONTRACTION_REDUCE_ERROR)
        local_reduce.grouped_tensors.update(self.grouped_tensors)
        return dataclasses.replace(outputs, output_contraction=self.contraction)


def canonical_output_contraction_source(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> torch.fx.Node:
    # [FUNCTION DOC] Starting at a contraction use's source, walk back through shape-
    # preserving pointwise nodes that have exactly one input depending on the GEMM,
    # stopping at the GEMM or when the chain is ambiguous. Returns the canonical node
    # used to compare provenance across uses.
    """Find one contraction use's GEMM provenance through pointwise wrappers."""
    while node is not gemm and is_shape_preserving_pointwise_node(node):
        inputs = [
            arg
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if local_reduce.graph.depends_on(arg, gemm)
        ]
        if len(inputs) != 1:
            break
        node = inputs[0]
    return node


def match_output_contraction_use(
    node: torch.fx.Node,
    gemm: torch.fx.Node,
    gemm_shape: tuple[Any, ...] | None,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> OutputContractionUse | None:
    # [FUNCTION DOC] Try to match a node as one supported use of grouped GEMM values.
    # Two forms: a getitem of a split along the last dimension of a 2D GEMM-shaped
    # tensor (chunked, group = columns / split size), or a select over a 3D view of the
    # GEMM output, either on dim 1 (chunked) or on the last dim of an already grouped
    # axis-1 layout (not chunked). Returns an OutputContractionUse, or None when shapes,
    # dims or sizes do not fit (group must exceed 1).
    """Match one supported use of grouped GEMM values in the main output."""
    if gemm_shape is None:
        return None
    normalized = local_reduce.graph.normalized_nodes.get(node)

    if isinstance(normalized, NormalizedGetItem):
        split = normalized.source
        split_normalized = local_reduce.graph.normalized_nodes.get(split)
        if not isinstance(split_normalized, NormalizedSplit):
            return None
        shape = tensor_meta_shape(split_normalized.source)
        if (
            shape is None
            or len(shape) != 2
            or not statically_known_shape_equal(shape, gemm_shape)
            or not local_reduce.graph.depends_on(split_normalized.source, gemm)
            or not isinstance(shape[-1], int)
            or split_normalized.dim not in (-1, 1)
        ):
            return None
        split_size = guarded_int(split_normalized.split_size)
        if split_size is None or split_size <= 0 or shape[-1] % split_size != 0:
            return None
        group = shape[-1] // split_size
        if group <= 1:
            return None
        return OutputContractionUse(
            source=split_normalized.source,
            group=group,
            chunked=True,
            group_indices=(normalized.index,),
            layout_node=split,
        )

    if not isinstance(normalized, NormalizedSelect):
        return None
    view = normalized.source
    view_normalized = local_reduce.graph.normalized_nodes.get(view)
    shape = tensor_meta_shape(view)
    if (
        not isinstance(view_normalized, NormalizedView)
        or not local_reduce.graph.depends_on(view_normalized.source, gemm)
        or shape is None
        or len(shape) != 3
        or not statically_known_shape_equal((shape[0], shape[1] * shape[2]), gemm_shape)
        or not -len(shape) <= normalized.dim < len(shape)
    ):
        return None
    index = guarded_int(normalized.index)
    if index is None:
        return None
    dim = normalized.dim % len(shape)
    if dim == 1:
        group = guarded_int(shape[1])
        if group is None or group <= 1:
            return None
        chunked = True
        layout_node = view
    elif dim == len(shape) - 1:
        grouped_layout = local_reduce.grouped_tensors.get(view)
        if grouped_layout is None or grouped_layout.axis != 1:
            return None
        group = grouped_layout.group
        chunked = False
        layout_node = None
    else:
        return None
    return OutputContractionUse(
        source=view_normalized.source,
        group=group,
        chunked=chunked,
        group_indices=(index,),
        layout_node=layout_node,
    )


def collect_output_contraction_uses(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> list[tuple[torch.fx.Node, OutputContractionUse]] | None:
    # [FUNCTION DOC] Depth-first walk from the main output over FX inputs, collecting
    # (node, use) pairs for matched contraction uses without mutating analysis state.
    # Returns None if the GEMM or a GEMM-dependent grouped tensor is reached outside a
    # matched use, or if no use is found.
    """Collect contraction uses without mutating analysis state."""
    gemm_shape = tensor_meta_shape(gemm)
    uses: list[tuple[torch.fx.Node, OutputContractionUse]] = []
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    stack: list[Any] = [output]
    while stack:
        node = stack.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        match = match_output_contraction_use(node, gemm, gemm_shape, local_reduce)
        if match is not None:
            uses.append((node, match))
            continue
        if node is gemm or (
            node in local_reduce.grouped_tensors
            and local_reduce.graph.depends_on(node, gemm)
        ):
            return None
        stack.extend(reversed(tuple(iter_fx_node_inputs((node.args, node.kwargs)))))
    return uses or None


def build_output_contraction_plan(
    output: torch.fx.Node,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> OutputContractionPlan | None:
    # [FUNCTION DOC] Build an OutputContractionPlan if all collected uses share one
    # canonical source, group count and chunked flag, cover every group index exactly
    # (negative indices are normalized), and the output shape equals GEMM rows by
    # columns divided by group. Returns None when any condition fails; raises
    # NotImplementedError if the covering uses produce an output of the wrong static
    # shape. Pure with respect to analysis state.
    """Build a plan when all grouped GEMM values form one output contraction."""
    collected = collect_output_contraction_uses(output, gemm, local_reduce)
    if collected is None:
        return None
    first = collected[0][1]
    source = canonical_output_contraction_source(first.source, gemm, local_reduce)
    group_indices: OrderedSet[int] = OrderedSet()
    select_indices: dict[torch.fx.Node, int] = {}
    grouped_tensors: dict[torch.fx.Node, FlexGemmLocalReduceGeometry] = {}
    for node, match in collected:
        if (
            canonical_output_contraction_source(match.source, gemm, local_reduce)
            is not source
            or match.group != first.group
            or match.chunked != first.chunked
        ):
            return None
        for index in match.group_indices:
            if not -first.group <= index < first.group:
                return None
            group_indices.add(index % first.group)
        if isinstance(local_reduce.graph.normalized_nodes.get(node), NormalizedSelect):
            select_indices[node] = match.group_indices[0] % first.group
        if match.layout_node is not None:
            grouped_tensors[match.layout_node] = FlexGemmLocalReduceGeometry(
                group=match.group, axis=1
            )
    if group_indices != OrderedSet(range(first.group)):
        return None
    gemm_meta = gemm.meta.get("val")
    output_meta = output.meta.get("val")
    if gemm_meta is None or output_meta is None or len(gemm_meta.shape) != 2:
        return None
    expected_shape = (gemm_meta.shape[0], gemm_meta.shape[1] // first.group)
    if not statically_known_shape_equal(output_meta.shape, expected_shape):
        raise NotImplementedError(FLEX_GEMM_OUTPUT_CONTRACTION_SHAPE_ERROR)
    return OutputContractionPlan(
        FlexGemmOutputContraction(group=first.group, chunked=first.chunked),
        select_indices,
        grouped_tensors,
    )


def validate_shape_preserving_main_output(
    outputs: FlexGemmOutputPlan,
    gemm: torch.fx.Node,
    local_reduce: FlexGemmLocalReduceAnalysis,
) -> None:
    # [FUNCTION DOC] Check a non-contraction main output. Raises NotImplementedError if
    # any select over an axis-1 grouped tensor remains (incomplete contraction) or if
    # the output shape is unknown or not statically equal to the GEMM shape. Returns
    # nothing.
    """Reject incomplete grouped selects and other shape-changing main outputs."""
    # TODO: Consider DCE before analysis if dead grouped selects become common.
    if any(
        isinstance(normalized, NormalizedSelect)
        and normalized.source in local_reduce.grouped_tensors
        and local_reduce.grouped_tensors[normalized.source].axis == 1
        for normalized in local_reduce.graph.normalized_nodes.values()
    ):
        raise NotImplementedError(
            "FlexGEMM grouped selects must form a complete output contraction"
        )
    output_shape = tensor_meta_shape(outputs.output)
    gemm_shape = tensor_meta_shape(gemm)
    if (
        output_shape is None
        or gemm_shape is None
        or not statically_known_shape_equal(output_shape, gemm_shape)
    ):
        raise NotImplementedError(FLEX_GEMM_MAIN_OUTPUT_SHAPE_ERROR)


@dataclasses.dataclass(frozen=True)
class FlexGemmEpilogueAnalysis:
    # [CLASS DOC] Frozen result of analysis shared by config selection, lowering and
    # emission: the GEMM node, the output plan, the local-reduce analysis, and select-
    # node to group-index mapping from the committed contraction (empty if none). Build
    # with from_graph_module; properties expose required grouped geometries and the
    # reduction plan.
    """Bundle the immutable analysis consumed by FlexGEMM lowering and emission.

    Attributes:
        gemm: The validated GEMM node shared by lowering and emission.
        outputs: Classification of main, auxiliary, and local-reduction outputs.
        local_reduce: Grouped layouts and local-reduction matches from the FX graph.
        output_contraction_select_indices: Select indices committed by the contraction plan.
    """

    gemm: torch.fx.Node
    outputs: FlexGemmOutputPlan
    local_reduce: FlexGemmLocalReduceAnalysis
    output_contraction_select_indices: dict[torch.fx.Node, int] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule, gemm: torch.fx.Node
    ) -> "FlexGemmEpilogueAnalysis":
        # [FUNCTION DOC] Run local-reduce analysis on the graph, classify outputs, bind
        # terminal dtype-view storage, then either commit an output contraction plan or
        # validate that the main output preserves the GEMM shape. Returns the assembled
        # analysis; propagates NotImplementedError from the steps.
        """Analyze grouped values and classify logical output consumers."""
        local_reduce = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module)
        outputs = bind_terminal_output_storage(
            local_reduce.graph, output_plan(graph_module, local_reduce)
        )
        contraction_plan = build_output_contraction_plan(
            outputs.output, gemm, local_reduce
        )
        if contraction_plan is None:
            validate_shape_preserving_main_output(outputs, gemm, local_reduce)
            output_contraction_select_indices = {}
        else:
            outputs = contraction_plan.commit(outputs, local_reduce)
            output_contraction_select_indices = contraction_plan.select_indices
        return cls(gemm, outputs, local_reduce, output_contraction_select_indices)

    @property
    def required_geometries(self) -> tuple[FlexGemmLocalReduceGeometry, ...]:
        # [FUNCTION DOC] Collect the distinct grouped geometries that constrain kernel
        # configuration: those of all matches, of the output local-reduce plan if any,
        # and an axis-1 geometry for the output contraction if any. Returns them as a
        # tuple.
        """Return every grouped geometry that constrains kernel configuration."""
        geometries = OrderedSet(
            match.geometry for match in self.local_reduce.matches.values()
        )
        if self.outputs.local_reduce is not None:
            geometries.add(self.outputs.local_reduce.match.geometry)
        if self.outputs.output_contraction is not None:
            geometries.add(
                FlexGemmLocalReduceGeometry(
                    group=self.outputs.output_contraction.group,
                    axis=1,
                )
            )
        return tuple(geometries)

    @property
    def reduction_plan(self) -> GemmReductionPlan | None:
        # [FUNCTION DOC] Return the backend-neutral reduction plan from the output plan
        # (may be None).
        """Return the backend-neutral reduction plan produced by FX analysis."""
        return self.outputs.reduction_plan


def analyze_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm: torch.fx.Node,
) -> FlexGemmEpilogueAnalysis:
    # [FUNCTION DOC] Analysis entry point called by FlexGEMM lowering: delegate to
    # FlexGemmEpilogueAnalysis.from_graph_module for the given body graph module and
    # GEMM node. Returns the immutable analysis used by later phases and by the
    # materialization step.
    # KNOWN ISSUE: docstring omits the gemm argument.
    """Analyze FlexGEMM body for output planning and epilogue code generation.

    This is the analysis entry point called by FlexGEMM lowering. It builds a
    dependency index, performs topological local-reduction analysis, and
    returns the shared immutable plan consumed by config selection and
    ``materialize_flex_gemm_epilogue``.

    Args:
        graph_module: FlexGEMM body graph containing GEMM and epilogue nodes.

    Returns:
        Output and local-reduction analysis shared by later lowering phases.
    """
    return FlexGemmEpilogueAnalysis.from_graph_module(graph_module, gemm)


def gemm_node(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> torch.fx.Node:
    # [FUNCTION DOC] Return the unique call_function node whose target is gemm_op in the
    # graph. Raises NotImplementedError unless exactly one such node exists.
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == gemm_op
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one GEMM body")
    return gemm_nodes[0]


class FlexGemmEpilogueEmitter:
    """Visit an analyzed FlexGEMM FX graph and emit its CuTeDSL epilogue.
    # [CLASS DOC] Visitor that lowers an analyzed FlexGEMM FX graph into CuTeDSL
    # epilogue source. It owns all mutable emission state: env (FX node to generated
    # value, seeded with the accumulator for the GEMM), grouped layouts, active grouped
    # layouts, store_sources (compressed local-reduce store expressions),
    # physical_reductions (combine/finalize callbacks), and the feed_main / aux /
    # feed_main_input nodes derived from the output local-reduce plan. Key methods:
    # lower_graph (topological traversal), lower_call_function (ordered handlers),
    # render (source text), materialize (full pipeline under virtualized handlers).

    The analysis dataclasses flow into each other as follows:

    ::

        FlexGemmEpilogueGraph
          `--> FlexGemmLocalReduceAnalysis
                 +--> grouped_tensors
                 `--> matches
                        `--> FlexGemmLocalReduceMatch
                               `--> FlexGemmOutputLocalReducePlan
                                      `--> optional FlexGemmLocalReduceStore

        FlexGemmLocalReduceAnalysis
          `--> output_plan()
                 `--> FlexGemmOutputPlan

        FlexGemmLocalReduceAnalysis + FlexGemmOutputPlan
          `--> FlexGemmEpilogueAnalysis
                 `--> FlexGemmEpilogueEmitter

    At emitter construction, ``analysis.outputs`` becomes ``self.outputs``;
    its local-reduce match and optional store initialize ``self.feed_main`` and
    ``self.aux``. ``analysis.local_reduce.grouped_tensors`` is copied into
    mutable emission state, while ``analysis.required_geometries`` determines
    the active grouped layouts.

    The emitter owns all mutable code-generation state: FX values lowered so far,
    grouped TensorSSA layouts, compressed-store expressions, and physical
    reduction callbacks. ``lower_graph`` performs a topological traversal and
    delegates each ``call_function`` node to ordered handlers; ``render`` turns
    the resulting state into the generated epilogue and callback source.
    """

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        analysis: FlexGemmEpilogueAnalysis,
        epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
        *,
        fast_math: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initialize emission state from the analysis: copy grouped
        # tensor geometry into emitter layouts, compute active layouts from required
        # geometries, create the kernel and the environment with the accumulator
        # variable. If the output plan is a feed-main local reduce, record the reduction
        # node, its input and the optional aux store node (raises AssertionError if the
        # node is not a normalized reduction); if only a store is planned, record the
        # aux node.
        self.graph_module = graph_module
        self.epilogue_arg_placeholders = epilogue_arg_placeholders
        self.fast_math = fast_math
        self.gemm = analysis.gemm
        self.graph = analysis.local_reduce.graph
        self.outputs = analysis.outputs
        self.output_contraction_select_indices = (
            analysis.output_contraction_select_indices
        )
        self.kernel = FlexGemmCuteDSLKernel()
        self.env: dict[torch.fx.Node, Any] = {
            self.gemm: CuteDSLCSEVariable(
                "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
            )
        }
        self.grouped_tensors = {
            node: GroupedTensorSSALayout(group=layout.group, axis=layout.axis)
            for node, layout in analysis.local_reduce.grouped_tensors.items()
        }
        self.active_grouped_layouts = OrderedSet(
            GroupedTensorSSALayout(group=geometry.group, axis=geometry.axis)
            for geometry in analysis.required_geometries
        )
        self.store_sources: dict[torch.fx.Node, Any] = {}
        self.physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction] = {}
        self.local_reduce = self.outputs.local_reduce
        self.feed_main: torch.fx.Node | None = None
        self.aux: torch.fx.Node | None = None
        self.feed_main_input: torch.fx.Node | None = None
        match self.local_reduce:
            case FlexGemmOutputLocalReducePlan(
                match=local_reduce_match, store=store, feeds_main=True
            ):
                self.feed_main = local_reduce_match.value_node
                reduction = self.graph.normalized_nodes.get(
                    local_reduce_match.value_node
                )
                if not isinstance(reduction, NormalizedReduction):
                    raise AssertionError("feed-main plans require a matched reduction")
                self.feed_main_input = reduction.source
                self.aux = None if store is None else store.node
            case FlexGemmOutputLocalReducePlan(
                store=FlexGemmLocalReduceStore(node=store_node)
            ):
                self.aux = store_node
            case None:
                pass

    def bind_epilogue_args(self) -> None:
        # [FUNCTION DOC] For each captured tensor placeholder create a named aux
        # parameter variable in env (bool is stored as uint8) and, when the logical
        # compute dtype differs from the physical dtype, wrap it in a conversion. Marks
        # variables scalar-valued when an output contraction is active. Mutates env.
        """Bind captured tensor placeholders to generated CuTeDSL parameters."""
        for index, node in enumerate(self.epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            self.env[node] = CuteDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
                is_scalar_expr=self.outputs.output_contraction is not None,
            )
            if logical_dtype != physical_dtype:
                self.env[node] = FlexGemmCuteDSLOpOverrides.to_dtype(
                    self.env[node], logical_dtype, use_compute_types=False
                )

    def bind_reduction(self, node: torch.fx.Node, lowered_reduce: Any) -> None:
        # [FUNCTION DOC] Store the lowered reduction for node in env. If node is the
        # feed-main reduction, instead bind it to the feed-main argument variable (same
        # dtype and shape) and inherit the grouped layout of its input; the generated
        # reduction is discarded.
        """Bind a generated reduction or replace it with the feed-main parameter."""
        if self.feed_main is not None and node is self.feed_main:
            self.env[node] = CuteDSLCSEVariable(
                LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                ValueRanges.unknown(),
                dtype=lowered_reduce.dtype,
                shape=lowered_reduce.shape,
            )
            if self.feed_main_input in self.grouped_tensors:
                self.grouped_tensors[node] = self.grouped_tensors[self.feed_main_input]
            return
        self.env[node] = lowered_reduce

    def lower_compact_inline_asm(self, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] Special case for an inline asm elementwise call applied to a
        # compact (reduced) value, directly or through one shape-preserving pointwise
        # node. Applies the call to the compact value, then broadcasts it back and
        # records the result in store_sources. Returns True if handled (env updated);
        # False when not applicable (feed-main active, wrong op, multiple tensor inputs,
        # no reduction source, or no grouped layout).
        """Apply inline asm before broadcasting a compact reduction value."""
        if (
            self.feed_main is not None
            or _cute_op_name(node.target) != "inline_asm_elementwise"
            or len(node.args) != 1
        ):
            return False
        tensor_inputs = [
            input_node
            for input_node in node.all_input_nodes
            if tensor_meta_shape(input_node) is not None
        ]
        if len(tensor_inputs) != 1:
            return False
        source = tensor_inputs[0]
        normalized = self.graph.normalized_nodes.get(source)
        compact_source = _cute_arg(source, self.env)
        if not isinstance(normalized, NormalizedReduction):
            if not is_shape_preserving_pointwise_node(source):
                return False
            source_inputs = [
                input_node
                for input_node in source.all_input_nodes
                if tensor_meta_shape(input_node) is not None
            ]
            if len(source_inputs) != 1:
                return False
            normalized = self.graph.normalized_nodes.get(source_inputs[0])
            if not isinstance(normalized, NormalizedReduction):
                return False
            compact_source = _cute_call(
                source,
                tuple(_cute_arg(arg, self.env) for arg in source.args),
                {
                    key: _cute_arg(value, self.env)
                    for key, value in source.kwargs.items()
                },
            )
        reduction_input = normalized.source
        layout = self.grouped_tensors.get(reduction_input)
        if layout is None:
            return False
        self.env[node] = _cute_call(
            node,
            (compact_source,),
            {key: _cute_arg(value, self.env) for key, value in node.kwargs.items()},
        )
        _, self.store_sources[node] = _keepdim_and_broadcast(
            self.kernel,
            self.env[node],
            layout,
            _cute_arg(reduction_input, self.env),
        )
        return True

    def lower_pointwise_store(self, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] Lower a shape-preserving pointwise node that consumes a
        # compressed store value by substituting store expressions for its arguments.
        # Updates env and store_sources and returns True; returns False when feed-main
        # is active, the node is not pointwise, or no input is in store_sources.
        """Lower pointwise expressions that consume a compressed store value."""
        if (
            self.feed_main is not None
            or not is_shape_preserving_pointwise_node(node)
            or not any(
                arg in self.store_sources
                for arg in iter_fx_node_inputs((node.args, tuple(node.kwargs.values())))
            )
        ):
            return False
        store_args = tuple(
            _local_reduce_store_arg(arg, self.env, self.store_sources)
            for arg in node.args
        )
        store_kwargs = {
            key: _local_reduce_store_arg(value, self.env, self.store_sources)
            for key, value in node.kwargs.items()
        }
        self.env[node] = _cute_call(node, store_args, store_kwargs)
        self.store_sources[node] = self.env[node]
        return True

    def propagate_physical_reduction(self, node: torch.fx.Node, source: Any) -> None:
        # [FUNCTION DOC] HELPER: If source has a physical reduction recorded, record the
        # same one for node so shape-only wrappers keep provenance. Mutates
        # physical_reductions.
        """Preserve physical callback provenance through shape-only wrappers."""
        if isinstance(source, torch.fx.Node) and source in self.physical_reductions:
            self.physical_reductions[node] = self.physical_reductions[source]

    def physical_finalize_arg(self, value: Any) -> Any:
        # [FUNCTION DOC] HELPER: Recursively convert a value into finalizer text: nodes
        # with a physical reduction become that reduction's finalize expression, tuples
        # and lists are converted elementwise (same type), anything else goes through
        # the normal argument conversion.
        """Replace physical reduction inputs with their generated value expression."""
        if isinstance(value, torch.fx.Node) and value in self.physical_reductions:
            return self.physical_reductions[value].finalize_expr
        if isinstance(value, (tuple, list)):
            return type(value)(self.physical_finalize_arg(item) for item in value)
        return _cute_arg(value, self.env)

    def compose_physical_finalize(self, node: torch.fx.Node) -> Any | None:
        # [FUNCTION DOC] Fold a pointwise node that consumes a physical reduction into
        # that reduction's finalizer. Returns None if no input is a physical reduction;
        # raises NotImplementedError if several distinct ones are consumed or the
        # combined expression is not a string. Side effects: copies the base store
        # source to node and records an updated physical reduction. Returns the new
        # finalize expression.
        """Fold a pointwise consumer into one generated physical finalizer."""
        physical_inputs = list(
            OrderedSet(
                arg
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in self.physical_reductions
            )
        )
        if not physical_inputs:
            return None
        if len(physical_inputs) > 1:
            raise NotImplementedError(LOCAL_REDUCE_SINGLE_PHYSICAL_FINALIZE_ERROR)
        base = physical_inputs[0]
        args = tuple(self.physical_finalize_arg(arg) for arg in node.args)
        kwargs = {
            key: self.physical_finalize_arg(value) for key, value in node.kwargs.items()
        }
        finalize_expr = _cute_call(node, args, kwargs)
        if not isinstance(finalize_expr, str):
            raise NotImplementedError(LOCAL_REDUCE_FINALIZE_SCALAR_ONLY_ERROR)
        self.store_sources[node] = self.store_sources[base]
        self.physical_reductions[node] = dataclasses.replace(
            self.physical_reductions[base], finalize_expr=finalize_expr
        )
        return finalize_expr

    def lower_call_function(self, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] Lower one call_function node. Order: full-scalar constants;
        # then handlers by normalized kind (terminal dtype view, squeeze, grouped split,
        # contraction select, getitem, online softmax prepare, view/reshape, reduction,
        # unsupported reduction); then physical finalizer composition, compact inline
        # asm, compressed pointwise store; finally the generic CuTeDSL call. Writes the
        # result into env. Raises NotImplementedError for dtype views that are not
        # terminal main outputs, unsupported reductions, or pointwise use of a physical
        # reduction when there is no aux store.
        """Lower one call_function node using the ordered FlexGEMM handlers."""
        lowered = lower_full_scalar(node)
        if lowered is not None:
            self.env[node] = lowered
            return
        normalized = self.graph.normalized_nodes.get(node)
        match normalized:
            case NormalizedDtypeView():
                if (
                    node is not self.outputs.output
                    or normalized.source is not self.outputs.output_storage
                ):
                    raise NotImplementedError(
                        "FlexGEMM supports dtype views only as terminal same-width main outputs"
                    )
                self.env[node] = _cute_arg(normalized.source, self.env)
                return
            case NormalizedSqueeze():
                lowered = lower_squeeze(node, normalized, self.env, self.store_sources)
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedSplit():
                lowered = lower_grouped_n_split(
                    node, normalized, self.env, self.kernel, self.grouped_tensors
                )
                if lowered is not None:
                    self.env[node] = lowered
                    return
            case NormalizedSelect():
                index = self.output_contraction_select_indices.get(node)
                if index is not None:
                    self.env[node] = lower_grouped_n_select(
                        normalized, index, self.env, self.kernel
                    )
                    return
            case NormalizedGetItem():
                lowered = lower_getitem(node, normalized, self.env, self.store_sources)
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedPrepareSoftmax():
                self.env[node] = lower_prepare_softmax_online(
                    node,
                    normalized,
                    self.env,
                    self.kernel,
                    self.grouped_tensors,
                    self.store_sources,
                )
                return
            case NormalizedView():
                lowered = lower_view_or_reshape(
                    node,
                    normalized,
                    self.env,
                    self.kernel,
                    self.grouped_tensors,
                    self.active_grouped_layouts,
                    self.store_sources,
                    node is self.feed_main_input,
                )
                if lowered is not None:
                    self.env[node] = lowered
                    self.propagate_physical_reduction(node, normalized.source)
                    return
            case NormalizedReduction():
                self.bind_reduction(
                    node,
                    lower_tensorssa_reduce(
                        node,
                        normalized,
                        self.env,
                        self.kernel,
                        self.grouped_tensors,
                        self.store_sources,
                        self.physical_reductions,
                    ),
                )
                return
            case NormalizedUnsupportedReduction():
                raise local_reduce_unsupported_tensorssa_error(
                    normalized.target, value_only=True
                )
        is_shape_preserving = is_shape_preserving_pointwise_node(node)
        if is_shape_preserving and self.feed_main is None:
            if self.aux is None and any(
                arg in self.physical_reductions
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
            ):
                raise NotImplementedError(LOCAL_REDUCE_POST_POINTWISE_FINALIZE_ERROR)
            physical_finalize = self.compose_physical_finalize(node)
            if physical_finalize is not None:
                self.env[node] = physical_finalize
                return
        if self.lower_compact_inline_asm(node) or self.lower_pointwise_store(node):
            return
        node_args = tuple(_cute_arg(arg, self.env) for arg in node.args)
        node_kwargs = {
            key: _cute_arg(value, self.env) for key, value in node.kwargs.items()
        }
        self.env[node] = _cute_call(node, node_args, node_kwargs)

    def lower_graph(self) -> None:
        # [FUNCTION DOC] Lower all nodes in FX order, skipping the GEMM, placeholders,
        # outputs and nodes whose value is an integer or symbolic int. Each node is
        # lowered with it set as the current node. Raises NotImplementedError for non
        # call_function nodes.
        """Lower body nodes in FX topological order."""
        for node in self.graph_module.graph.nodes:
            if node is self.gemm or node.op in ("placeholder", "output"):
                continue
            if isinstance(node.meta.get("val"), (int, torch.SymInt)):
                continue
            with V.set_current_node(node):
                if node.op != "call_function":
                    raise NotImplementedError(
                        f"unsupported FlexGEMM epilogue node: {node.format_node()}"
                    )
                self.lower_call_function(node)

    @staticmethod
    def aux_result(
        aux: torch.fx.Node | None, store_sources: dict[torch.fx.Node, Any]
    ) -> Any | None:
        # [FUNCTION DOC] Return the compressed aux expression for aux from
        # store_sources, or None if aux is None. Raises NotImplementedError if an aux
        # was planned but no TensorSSA store expression exists.
        """Return the compressed-aux expression or reject missing TensorSSA."""
        if aux is None:
            return None
        result = store_sources.get(aux)
        if result is None:
            raise NotImplementedError(LOCAL_REDUCE_AUX_TENSORSSA_ERROR)
        return result

    def render(self) -> tuple[str, str]:
        # [FUNCTION DOC] Produce the final generated source. Builds the function body
        # from kernel lines, parameter list (acc, aux params, optional feed-main arg),
        # and a return of the main output (a tuple when aux outputs or a compressed aux
        # exist). Computes a short sha256-based name from inline asm key, fast-math
        # flag, graph code, body and result, and when a physical reduction exists emits
        # combine and finalize jit helper functions with fixed cache keys. Returns
        # (function name, full source text including imports).
        """Render the generated epilogue and physical callback source."""
        from torch._inductor.codegen.cutedsl._inline_asm import inline_asm_cache_key

        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        if body:
            body += "\n"
        aux_args = [
            f"aux{index}" for index in range(len(self.epilogue_arg_placeholders))
        ]
        feed_main_args = (
            [LOCAL_REDUCE_FEED_MAIN_ARG_NAME] if self.feed_main is not None else []
        )
        epilogue_params = ", ".join(["acc", *aux_args, *feed_main_args])
        result = _cute_arg(self.outputs.output, self.env)
        aux_result = self.aux_result(self.aux, self.store_sources)
        if self.outputs.aux_outputs or aux_result is not None:
            tuple_items = [result]
            tuple_items.extend(
                _cute_arg(aux_output, self.env)
                for aux_output in self.outputs.aux_outputs
            )
            if aux_result is not None:
                tuple_items.append(aux_result)
            result = f"({', '.join(str(item) for item in tuple_items)})"
        physical_reduction = (
            None
            if self.local_reduce is None
            else self.physical_reductions.get(self.local_reduce.match.value_node)
        )
        physical_reduction_payload = (
            ""
            if physical_reduction is None
            else (
                f"\ncombine {physical_reduction.combine_expr}"
                f"\nfinalize {physical_reduction.finalize_expr}"
            )
        )
        key_payload = (
            f"inline_asm={inline_asm_cache_key()}\n"
            f"fast_math={self.fast_math}\n{self.graph_module.code}\n"
            f"{body}\nreturn {result}{physical_reduction_payload}"
        )
        key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
        name = f"flex_gemm_epilogue_{key}"
        local_reduce_source = ""
        if physical_reduction is not None:
            combine_name = f"{name}{LOCAL_REDUCE_COMBINE_FN_SUFFIX}"
            finalize_name = f"{name}{LOCAL_REDUCE_FINALIZE_FN_SUFFIX}"
            local_reduce_source = (
                f"@cute.jit\ndef {combine_name}(lhs, rhs):\n"
                f"    return {physical_reduction.combine_expr}\n"
                f"{combine_name}.__cache_key__ = lambda: {combine_name!r}\n\n"
                f"@cute.jit\ndef {finalize_name}(value):\n"
                f"    return {physical_reduction.finalize_expr}\n"
                f"{finalize_name}.__cache_key__ = lambda: {finalize_name!r}\n\n"
            )
        return (
            name,
            "import cutlass\n"
            "import cutlass.cute as cute\n"
            "import operator\n"
            "from cutlass._mlir.dialects import math as mlir_math\n"
            "from cutlass._mlir_helpers import math as cutlass_math\n"
            "from torch._inductor.codegen.cutedsl._inline_asm import (\n"
            "    inline_asm_elementwise_intrinsic,\n"
            ")\n\n"
            f"{local_reduce_source}"
            f"@cute.jit\ndef {name}({epilogue_params}):\n"
            f"{body}    return {result}\n",
        )

    def materialize(self) -> tuple[str, str]:
        """Lower and render this epilogue under the CuTeDSL virtualized handlers."""
        # [FUNCTION DOC] Run the full pipeline: with the kernel and CuTeDSL op overrides
        # installed as virtualized handlers (and the fast-math setting applied), bind
        # epilogue args and lower the graph, then render. Returns (name, source).
        with (
            V.set_kernel_handler(self.kernel),
            V.set_ops_handler(FlexGemmCuteDSLOpOverrides()),
            use_cutedsl_fast_math(self.fast_math),
        ):
            self.bind_epilogue_args()
            self.lower_graph()
        return self.render()


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    analysis: FlexGemmEpilogueAnalysis,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
    *,
    fast_math: bool = False,
) -> tuple[str, str]:
    # [FUNCTION DOC] Code-generation entry point: construct a FlexGemmEpilogueEmitter
    # and materialize it. Args are the graph module, the analysis, captured tensor
    # placeholders exposed as aux parameters, and the fast_math flag. Returns (epilogue
    # function name, CuTeDSL source).
    # KNOWN ISSUE: docstring mentions a gemm_op argument that does not exist.
    """Materialize an analyzed FlexGEMM body as generated CuTeDSL source.

    This is the code-generation entry point called by FlexGEMM lowering after
    ``analyze_flex_gemm_epilogue`` has classified outputs and local-reduction
    matches. The emitter visits the FX graph once in topological order while
    owning the environment and reduction state needed across nodes.

    Args:
        graph_module: FlexGEMM body graph containing the GEMM and epilogue nodes.
        gemm_op: GEMM overload expected to occur exactly once in the body.
        analysis: Shared output and local-reduction analysis for the graph.
        epilogue_arg_placeholders: Captured tensor placeholders exposed as
            generated epilogue parameters.
        fast_math: Whether supported CuTeDSL math operations may use approximate
            fast-math lowering.

    Returns:
        The generated epilogue function name and complete CuTeDSL source.
    """
    return FlexGemmEpilogueEmitter(
        graph_module,
        analysis,
        epilogue_arg_placeholders,
        fast_math=fast_math,
    ).materialize()
