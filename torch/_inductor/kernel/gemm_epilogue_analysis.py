# mypy: allow-untyped-defs
# [MODULE DOC] FX-graph analysis for grouped GEMM epilogues: recognizes grouped reshapes of the
# 2-D GEMM output (group along M or N), propagates grouped layouts and local-reduction matches
# through pointwise ops, detects "feed-main" patterns where a reduction is broadcast back into the
# main output, and builds output plans (GemmOutputPlan and related records) that finalize into a
# GemmReductionPlan. Many unsupported patterns raise NotImplementedError with shared messages.
# No import-time side effects beyond constant tables.
"""Shared FX analysis and output planning for grouped GEMM epilogues."""

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR,
    LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MATCH_NODE_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    local_reduce_unsupported_tensorssa_error,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    is_shape_preserving_pointwise_node,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmEpilogueGraph,
    GemmReductionGeometry,
    GemmReductionPlan,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    guarded_int,
    normalize_shape,
    statically_known_equal,
    statically_known_shape_equal,
)
from torch.utils._ordered_set import OrderedSet


def _is_inferred_reshape_dim(value: Any) -> bool:
    # [FUNCTION DOC] HELPER: True only for a plain integer equal to -1 (the inferred-size marker).
    """Return whether a reshape dimension is the literal inferred-size marker."""
    return isinstance(value, int) and value == -1


def _kept_dim_matches_source(kept_size: Any, source_size: Any) -> bool:
    # [FUNCTION DOC] HELPER: True if the untouched reshape dimension is -1 or statically known equal to
    # the source dimension.
    return _is_inferred_reshape_dim(kept_size) or statically_known_equal(
        kept_size, source_size
    )


def _guard_grouped_reshape_group(
    shape: tuple[Any, ...], source_shape: tuple[Any, ...]
) -> tuple[Any, ...]:
    # [FUNCTION DOC] HELPER: For a 3-entry reshape shape, find a candidate group position (last entry
    # paired with the first as kept dim, or second-to-last paired with the last) whose kept dim
    # matches the source and whose group entry is symbolic (SymInt); replace that entry by its
    # guarded concrete int (side effect: may add a shape guard) and return the new tuple. Otherwise
    # return the shape unchanged.
    """Specialize a backed group dimension used to recognize a grouped reshape."""
    if len(shape) != 3:
        return shape
    for group_index, kept_index in ((-1, 0), (-2, -1)):
        if not _kept_dim_matches_source(shape[kept_index], source_shape[kept_index]):
            continue
        group_value = shape[group_index]
        symbolic = (
            group_value.meta.get("val")
            if isinstance(group_value, torch.fx.Node)
            else group_value
        )
        if not isinstance(symbolic, torch.SymInt):
            continue
        group = guarded_int(group_value)
        if group is not None:
            result = list(shape)
            result[group_index] = group
            return tuple(result)
    return shape


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GemmReductionGeometry | None:
    # [FUNCTION DOC] HELPER: Recognize grouped reshape syntax from the shape alone: rank 3 or 4 with a
    # positive int last entry preceded by -1 (axis 1, group is last), or a -1 three from the end
    # followed by a positive int (axis 0, group is second-to-last). Else None.
    """Match grouped-reshape syntax before validating source geometry."""
    if len(shape) not in (3, 4):
        return None
    if (
        isinstance(shape[-1], int)
        and shape[-1] > 0
        and _is_inferred_reshape_dim(shape[-2])
    ):
        return GemmReductionGeometry(group=shape[-1], axis=1)
    if (
        _is_inferred_reshape_dim(shape[-3])
        and isinstance(shape[-2], int)
        and shape[-2] > 0
    ):
        return GemmReductionGeometry(group=shape[-2], axis=0)
    return None


def _group_count_matches_selected_dim(
    group_count: Any,
    selected_size: Any,
    group: int,
    kept_size: Any,
) -> bool:
    # [FUNCTION DOC] HELPER: True if group_count is -1, or group_count times group equals the selected
    # source dimension, or (when kept_size is not -1) group_count equals the selected dimension
    # divided by group; comparisons are static (no guards).
    """Match a group count, allowing -1 to infer the selected source dimension."""
    if _is_inferred_reshape_dim(group_count):
        return True
    return statically_known_equal(group_count * group, selected_size) or (
        not _is_inferred_reshape_dim(kept_size)
        and statically_known_equal(group_count, selected_size // group)
    )


def _grouped_layout_matches_source_shape(
    shape: tuple[Any, ...],
    source_shape: tuple[Any, ...],
    layout: GemmReductionGeometry,
) -> bool:
    # [FUNCTION DOC] HELPER: Check that a 3-D reshape shape splits exactly one axis of the 2-D source
    # (M, N). Axis 1 expects (kept M, group count, group); axis 0 expects (group count, group, kept
    # N); the group must equal the layout's group and the other dims must match the source.
    # Returns False for non-3-D shapes or any mismatch.
    """Require a 2-D GEMM output reshape to split exactly M or N."""
    if len(shape) != 3:
        return False

    m, n = source_shape
    match layout.axis, shape:
        case 1, (kept_m, group_count, group) if group == layout.group:
            return _kept_dim_matches_source(
                kept_m, m
            ) and _group_count_matches_selected_dim(group_count, n, group, kept_m)
        case 0, (group_count, group, kept_n) if group == layout.group:
            return _kept_dim_matches_source(
                kept_n, n
            ) and _group_count_matches_selected_dim(group_count, m, group, kept_n)
        case _:
            return False


def grouped_tensor_layout(
    shape: Any, source_shape: Any | None = None
) -> GemmReductionGeometry | None:
    # [FUNCTION DOC] Recognize grouped M/N geometry of a reshape target shape. Normalizes the shape
    # (unwrapping a single nested sequence); non-tuples give None. With a 2-D source shape, may
    # specialize a symbolic group (guard side effect), tries axis 1 (group last) and axis 0 (group
    # second to last) candidates against the source, and returns the first that fits. If none fit
    # but the shape looks grouped syntactically, raises NotImplementedError (grouped reshape
    # error); otherwise None. Without a source shape, only syntactic recognition is used.
    """Recognize grouped M/N geometry, specializing backed group dimensions."""
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple):
        return None
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        shape = normalize_shape(shape[0])
    if source_shape is not None:
        source_shape = normalize_shape(source_shape)
        if isinstance(source_shape, tuple) and len(source_shape) == 2:
            shape = _guard_grouped_reshape_group(shape, source_shape)
            candidates = []
            match shape:
                case (*_, int(group)) if group > 0:
                    candidates.append(GemmReductionGeometry(group=group, axis=1))
            match shape:
                case (*_, int(group), _) if group > 0:
                    candidates.append(GemmReductionGeometry(group=group, axis=0))
            for layout in candidates:
                if _grouped_layout_matches_source_shape(shape, source_shape, layout):
                    return layout
            if _syntactic_grouped_tensor_layout(shape) is not None:
                raise NotImplementedError(LOCAL_REDUCE_GROUPED_RESHAPE_ERROR)
            return None
    return _syntactic_grouped_tensor_layout(shape)


FEED_MAIN_BINARY_FUNCTIONS = frozenset(
    (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
    )
)


@dataclasses.dataclass(frozen=True)
class GemmLocalReduceMatch:
    # [CLASS DOC] Immutable record of a supported grouped local reduction: value_node (node producing
    # the value), geometry, reduction_node (defaults to value_node) and reduction_type. Offers
    # conversion to an output plan and helpers that merge several matches.
    """Describe a supported grouped local-reduction value found in the FX graph.

    Attributes:
        value_node: FX node that produces the matched local-reduction value.
        geometry: Group size and GEMM output axis reduced by the value.
    """

    value_node: torch.fx.Node
    geometry: GemmReductionGeometry
    reduction_node: torch.fx.Node | None = None
    reduction_type: str | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises RuntimeError if value_node is not an FX node; defaults reduction_node
        # to value_node (mutating the frozen record via object setattr).
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)
        if self.reduction_node is None:
            object.__setattr__(self, "reduction_node", self.value_node)

    def to_plan(
        self,
        *,
        store: "GemmLocalReduceStore | None",
        feeds_main: bool,
    ) -> "GemmOutputLocalReducePlan":
        # [FUNCTION DOC] Bind this match to an optional compressed store and a feeds_main flag.
        """Bind this matched value to its output consumers."""
        return GemmOutputLocalReducePlan(self, store=store, feeds_main=feeds_main)

    @classmethod
    def common(
        cls,
        matches: list["GemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "GemmLocalReduceMatch | None":
        # [FUNCTION DOC] Return the first match if all matches share one geometry; None for an empty
        # list; raises NotImplementedError(mixed_match_error) if geometries differ.
        """Return the common match when all values use one reduction geometry."""
        if not matches:
            return None
        match = matches[0]
        if any(item.geometry != match.geometry for item in matches):
            raise NotImplementedError(mixed_match_error)
        return match

    @classmethod
    def common_value(
        cls,
        matches: list["GemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "GemmLocalReduceMatch | None":
        # [FUNCTION DOC] Like common, and additionally requires every match to have the identical
        # value_node; otherwise raises NotImplementedError (one-physical-value error).
        """Return the common match when all consumers use one physical value."""
        match = cls.common(matches, mixed_match_error)
        if match is None:
            return None
        if any(item.value_node is not match.value_node for item in matches):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        return match


@dataclasses.dataclass(frozen=True)
class GemmLocalReduceStore:
    # [CLASS DOC] Immutable record of where a compressed local reduction is returned: the output node
    # and its index among auxiliary outputs.
    """Describe where a compressed local reduction appears in graph outputs.

    Attributes:
        node: FX node returned as the compressed local-reduction output.
        aux_index: Position of that node among the graph's auxiliary outputs.
    """

    node: torch.fx.Node
    aux_index: int

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises RuntimeError if node is not an FX node or aux_index is negative.
        if not isinstance(self.node, torch.fx.Node) or self.aux_index < 0:
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)


@dataclasses.dataclass(frozen=True)
class GemmOutputLocalReducePlan:
    # [CLASS DOC] Immutable binding of a local-reduction match to its consumers: an optional
    # compressed store and/or a flag that the reduced value feeds the main output.
    """Bind a matched local reduction to store and/or main-output consumers.

    Attributes:
        match: Supported local-reduction value identified during FX analysis.
        store: Compressed auxiliary output receiving the value, when requested.
        feeds_main: Whether the reduced value is also consumed by the main output.
    """

    match: GemmLocalReduceMatch
    store: GemmLocalReduceStore | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises RuntimeError if match is not a GemmLocalReduceMatch or the plan has
        # neither a store nor feeds_main.
        if not isinstance(self.match, GemmLocalReduceMatch) or (
            self.store is None and not self.feeds_main
        ):
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)

    @property
    def needs_physical_callbacks(self) -> bool:
        # [FUNCTION DOC] Delegates to the matched geometry's needs_physical_callbacks.
        return self.match.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class GemmOutputPlan:
    # [CLASS DOC] Immutable classification of the values returned by a FlexGEMM body: main output
    # node, same-shape auxiliary output nodes, and an optional local-reduce plan.
    """Classify the values returned by a FlexGEMM body.

    Attributes:
        output: FX node returned as the main GEMM result.
        aux_outputs: Same-shape auxiliary FX outputs returned after the main result.
        local_reduce: Compressed or feed-main local-reduction output behavior.
    """

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: GemmOutputLocalReducePlan | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Raises RuntimeError unless output and all aux outputs are FX nodes.
        if not isinstance(self.output, torch.fx.Node) or not all(
            isinstance(aux_output, torch.fx.Node) for aux_output in self.aux_outputs
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)

    @property
    def reduction_plan(self) -> GemmReductionPlan | None:
        # [FUNCTION DOC] Convert the local-reduce plan into a GemmReductionPlan, or None if there is no
        # local reduce or no reduction type. The reduction output is the store node name (if any),
        # source type is identity, the primary output is the main output name, and feed_output is
        # the main output name only when the reduction feeds main.
        """Finalize FX ownership metadata into the shared reduction contract."""
        local_reduce = self.local_reduce
        if local_reduce is None:
            return None
        match = local_reduce.match
        reduction_type = match.reduction_type
        if reduction_type is None:
            return None
        reduction_output = (
            local_reduce.store.node.name if local_reduce.store is not None else None
        )
        return GemmReductionPlan(
            reduction_output,
            match.geometry.group,
            match.geometry.axis,
            reduction_type,
            "identity",
            self.output.name,
            feeds_main=local_reduce.feeds_main,
            feed_output=self.output.name if local_reduce.feeds_main else None,
        )


@dataclasses.dataclass
class GemmLocalReduceAnalysis:
    # [CLASS DOC] Mutable analysis state for one FX graph: graph (dependency and normalization
    # index), grouped_tensors (node to grouped geometry) and matches (node to local-reduce
    # match). Built by from_graph_module via a topological visit; planning methods then derive
    # compressed-aux and feed-main output plans. Raises NotImplementedError on unsupported shapes.
    """Collect grouped TensorSSA layouts and supported local-reduction matches.

    ``from_graph_module`` visits the FX graph in topological order. See
    ``GemmReductionGeometry`` for the grouped layout attached to reshape and
    pointwise nodes, and ``GemmLocalReduceMatch`` for each supported reduced
    value found from those layouts.

    Attributes:
        graph: Dependency index used by recursive feed-main matching.
        grouped_tensors: FX nodes whose values carry a grouped TensorSSA layout.
        matches: FX values matched to a supported grouped local reduction.
    """

    graph: GemmEpilogueGraph
    grouped_tensors: dict[torch.fx.Node, GemmReductionGeometry] = dataclasses.field(
        default_factory=dict
    )
    matches: dict[torch.fx.Node, GemmLocalReduceMatch] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "GemmLocalReduceAnalysis":
        # [FUNCTION DOC] Index the graph and visit nodes in order up to (excluding) the output node,
        # recording grouped layouts and matches. Returns the populated analysis.
        """Build shared dependency and reduction state in one topological pass."""
        nodes = tuple(graph_module.graph.nodes)
        analysis = cls(GemmEpilogueGraph.from_nodes(nodes))
        for node in nodes:
            if node.op == "output":
                break
            analysis.visit_node(node)
        return analysis

    def visit_node(self, node: torch.fx.Node) -> None:
        # [FUNCTION DOC] Process one call_function node: views propagate an existing match or bind a
        # new grouped layout; reductions and softmax prep bind grouped reductions; unsupported
        # reductions on grouped tensors raise NotImplementedError; squeeze and getitem propagate
        # matches; shape-preserving pointwise nodes propagate layouts and matches (mixed-match
        # error on conflicts). Mutates grouped_tensors and matches. Other node kinds are ignored.
        """Record grouped layouts and local-reduction matches for one FX node."""
        if node.op != "call_function":
            return
        normalized = self.graph.normalized_nodes.get(node)
        if isinstance(normalized, NormalizedView):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
            if self.bind_grouped_layout(node, normalized.shape, normalized.source):
                return
        elif isinstance(normalized, NormalizedReduction):
            if self.bind_grouped_reduction(
                node,
                normalized.source,
                normalized.dim,
                normalized.dtype,
                reduction_type=normalized.reduction_type,
            ):
                return
        elif isinstance(normalized, NormalizedPrepareSoftmax):
            if self.bind_grouped_reduction(
                node,
                normalized.source,
                normalized.dim,
                raise_invalid_dims=False,
            ):
                return
        elif isinstance(normalized, NormalizedUnsupportedReduction):
            if normalized.source in self.grouped_tensors:
                raise local_reduce_unsupported_tensorssa_error(normalized.target)
        elif isinstance(normalized, (NormalizedSqueeze, NormalizedGetItem)):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)

    def bind_grouped_layout(self, node: torch.fx.Node, shape: Any, source: Any) -> bool:
        # [FUNCTION DOC] If a reshape of source to shape forms a grouped layout (using the source tensor
        # shape when known), record it for node and return True; otherwise False. May raise
        # NotImplementedError from layout recognition.
        """Attach a grouped TensorSSA layout introduced by a reshape."""
        source_shape = (
            tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
        )
        grouped_layout = grouped_tensor_layout(shape, source_shape)
        if grouped_layout is None or not isinstance(source, torch.fx.Node):
            return False
        self.grouped_tensors[node] = grouped_layout
        return True

    def propagate_local_reduce_match(self, node: torch.fx.Node, source: Any) -> bool:
        # [FUNCTION DOC] If source is a node with a recorded match, record the same match for node and
        # return True; else False.
        """Copy a matched local-reduction value through an FX wrapper."""
        if not isinstance(source, torch.fx.Node):
            return False
        match = self.matches.get(source)
        if match is None:
            return False
        self.matches[node] = match
        return True

    def bind_grouped_reduction(
        self,
        node: torch.fx.Node,
        input_node: Any,
        dim: Any,
        dtype: Any = None,
        *,
        reduction_type: str | None = None,
        raise_invalid_dims: bool = True,
    ) -> bool:
        # [FUNCTION DOC] If input_node has a grouped layout, record a match for node when the reduced dim
        # is the grouped one. Returns False when the input is not a grouped node. Raises
        # NotImplementedError for an explicit dtype, for an unsupported group size (validator),
        # and for a non-grouped reduce dim unless raise_invalid_dims is false (then returns False).
        # Side effect: stores the match with reduction_type.
        """Match and record a reduction over a grouped TensorSSA layout."""
        if not isinstance(input_node, torch.fx.Node):
            return False
        layout = self.grouped_tensors.get(input_node)
        if layout is None:
            return False
        if dtype is not None:
            raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group)
        if not layout.matches_reduction_dim(dim):
            if not raise_invalid_dims:
                return False
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        self.matches[node] = GemmLocalReduceMatch(
            node,
            layout,
            reduction_node=node,
            reduction_type=reduction_type,
        )
        return True

    def has_physical_grouped_input(self, value: Any) -> bool:
        # [FUNCTION DOC] True if any node in value, or any of its dependencies, is a grouped tensor whose
        # layout requires physical callbacks and whose geometry is used by some active match.
        """Return whether a value depends on a grouped layout needing callbacks."""
        active_geometries = OrderedSet(
            match.geometry for match in self.matches.values()
        )
        physical_grouped_nodes = OrderedSet(
            node
            for node, layout in self.grouped_tensors.items()
            if layout.needs_physical_callbacks and layout in active_geometries
        )
        return any(
            node in physical_grouped_nodes
            or any(
                dependency in physical_grouped_nodes
                for dependency in self.graph.dependencies.get(node, ())
            )
            for node in iter_fx_node_inputs(value)
        )

    def propagate_pointwise_match(
        self, node: torch.fx.Node, mixed_match_error: str
    ) -> bool:
        # [FUNCTION DOC] Propagate grouped layouts from a pointwise node's inputs to the node (raises
        # NotImplementedError if inputs carry different layouts), then propagate input matches
        # (common match, else raises mixed_match_error) with value_node set to this node. Returns
        # True when a match was recorded.
        """Propagate grouped layouts and local-reduction matches through pointwise ops."""
        grouped_layouts = [
            self.grouped_tensors[arg]
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if arg in self.grouped_tensors
        ]
        if grouped_layouts:
            grouped_layout = grouped_layouts[0]
            if any(layout != grouped_layout for layout in grouped_layouts):
                raise NotImplementedError(LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR)
            self.grouped_tensors[node] = grouped_layout
        match = GemmLocalReduceMatch.common(
            [
                self.matches[arg]
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in self.matches
            ],
            mixed_match_error,
        )
        if match is None:
            return False
        self.matches[node] = dataclasses.replace(match, value_node=node)
        return True

    def match_feed_value(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Recursively find the single grouped reduction (of grouped_source, keepdim,
        # no dtype, matching reduce dim) that produces a broadcast value, looking through
        # shape-preserving pointwise nodes. Returns None if value is not a node or not such a
        # pattern. Raises NotImplementedError when a reduction uses another source (source
        # expression error if derived from grouped_source) or violates the constraints, or when
        # different reduction values are mixed.
        """Find the grouped reduction that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            if normalized.source is not grouped_source:
                if self.graph.depends_on(normalized.source, grouped_source):
                    raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                normalized.dtype is not None
                or not normalized.keepdim
                or not layout.matches_reduction_dim(normalized.dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            return GemmLocalReduceMatch(
                value,
                layout,
                reduction_node=value,
                reduction_type=normalized.reduction_type,
            )
        if not is_shape_preserving_pointwise_node(value):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
            if (match := self.match_feed_value(arg, grouped_source, layout)) is not None
        ]
        return GemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
        )

    def validate_hidden_feed_main_reduction_input(
        self,
        input_node: Any,
        grouped_source: torch.fx.Node,
    ) -> None:
        # [FUNCTION DOC] Raise NotImplementedError if input_node is grouped_source itself, depends on it,
        # or has a physical grouped input; non-node inputs are accepted.
        """Reject reduction inputs that would need another physical feed-main value."""
        if input_node is grouped_source:
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        if not isinstance(input_node, torch.fx.Node):
            return
        if self.graph.depends_on(input_node, grouped_source):
            raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
        if self.has_physical_grouped_input(input_node):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)

    def validate_feed_main_source_reductions(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        selected_reduction: torch.fx.Node,
        seen: OrderedSet[torch.fx.Node] | None = None,
    ) -> None:
        # [FUNCTION DOC] Walk value's input graph (each node once via seen), skipping the selected
        # reduction, and validate the source of every other reduction found with
        # validate_hidden_feed_main_reduction_input. Raises NotImplementedError on violations.
        """Reject hidden physical reductions outside the selected feed-main value."""
        if not isinstance(value, torch.fx.Node):
            for arg in iter_fx_node_inputs(value):
                self.validate_feed_main_source_reductions(
                    arg, grouped_source, selected_reduction, seen
                )
            return
        if value is selected_reduction:
            return
        if seen is None:
            seen = OrderedSet()
        if value in seen:
            return
        seen.add(value)
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            self.validate_hidden_feed_main_reduction_input(
                normalized.source, grouped_source
            )
        for arg in iter_fx_node_inputs((value.args, value.kwargs)):
            self.validate_feed_main_source_reductions(
                arg, grouped_source, selected_reduction, seen
            )

    def validate_feed_main_source_match(
        self,
        source: torch.fx.Node,
        match: GemmLocalReduceMatch | None,
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Pass through a None match; if the match value is a reduction, validate the
        # source expression for hidden reductions (may raise NotImplementedError) and return the match.
        """Preserve the one-physical-value ABI across recursive source matching."""
        if match is None:
            return None
        normalized = self.graph.normalized_nodes.get(match.value_node)
        if isinstance(normalized, NormalizedReduction):
            self.validate_feed_main_source_reductions(
                source, normalized.source, match.value_node
            )
        return match

    @staticmethod
    def feed_main_binary_candidates(
        source: torch.fx.Node,
    ) -> tuple[tuple[Any, Any], ...]:
        # [FUNCTION DOC] For a binary add/sub/mul/div call, return both operand orderings as pairs
        # (lhs, rhs) and (rhs, lhs); otherwise an empty tuple.
        """Return operand orderings for supported binary feed-main expressions."""
        if (
            len(source.args) < 2
            or source.op != "call_function"
            or source.target not in FEED_MAIN_BINARY_FUNCTIONS
        ):
            return ()
        lhs, rhs = source.args[:2]
        return ((lhs, rhs), (rhs, lhs))

    def feed_main_grouped_reduction(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> bool:
        # [FUNCTION DOC] True if value is, or via shape-preserving pointwise nodes contains, a keepdim
        # reduction without dtype over a matching dim whose source is or depends on grouped_source.
        """Return whether a candidate contains a grouped feed-main reduction."""
        if not isinstance(value, torch.fx.Node):
            return False
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            return (
                normalized.dtype is None
                and bool(normalized.keepdim)
                and layout.matches_reduction_dim(normalized.dim)
                and (
                    normalized.source is grouped_source
                    or self.graph.depends_on(normalized.source, grouped_source)
                )
            )
        if not is_shape_preserving_pointwise_node(value):
            return False
        return any(
            self.feed_main_grouped_reduction(arg, grouped_source, layout)
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
        )

    def match_feed_main_candidate(
        self,
        grouped_source: Any,
        value: Any,
        output_meta: Any,
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Try one operand ordering. grouped_source must be a view node with a grouped
        # layout. For axis 1, return None (axis-1 groups within one fragment lower as plain
        # TensorSSA) when a grouped reduction exists and the group fits the fragment width, or
        # when none exists; a larger group raises NotImplementedError. For axis 0, validates
        # feed-main capability, returns None if output and source shapes differ, else delegates
        # to match_feed_value.
        """Match one grouped-source and reduced-value operand ordering."""
        if not isinstance(grouped_source, torch.fx.Node) or not isinstance(
            value, torch.fx.Node
        ):
            return None
        normalized = self.graph.normalized_nodes.get(grouped_source)
        if not isinstance(normalized, NormalizedView):
            return None
        source_node = normalized.source
        layout = self.grouped_tensors.get(grouped_source)
        if layout is None:
            return None
        if layout.axis != 0:
            if not self.feed_main_grouped_reduction(value, grouped_source, layout):
                return None
            if layout.group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
                # Intentional fallthrough: axis-1 feeds within one TensorSSA
                # fragment lower as plain generated TensorSSA without a feed plan.
                return None
            raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR)
        validate_local_reduce_feed_main_capability(layout.axis, layout.group)
        source_meta = source_node.meta.get("val")
        if (
            output_meta is not None
            and source_meta is not None
            and not statically_known_shape_equal(output_meta.shape, source_meta.shape)
        ):
            return None
        return self.match_feed_value(value, grouped_source, layout)

    def match_feed_main_source(
        self,
        source: torch.fx.Node,
        output_meta: Any,
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Search a pointwise expression for the single physical feed-main value: try binary
        # operand orderings first, else recurse into inputs of shape-preserving pointwise nodes.
        # Matches must share one value node (else NotImplementedError). The result is validated
        # for hidden physical reductions; None when nothing matches.
        """Find one physical feed-main value inside a pointwise expression."""
        matches = [
            match
            for grouped_source, value in self.feed_main_binary_candidates(source)
            if (
                match := self.match_feed_main_candidate(
                    grouped_source, value, output_meta
                )
            )
            is not None
        ]
        if matches:
            return self.validate_feed_main_source_match(
                source,
                GemmLocalReduceMatch.common_value(
                    matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
                ),
            )
        if not is_shape_preserving_pointwise_node(source):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((source.args, source.kwargs))
            if isinstance(arg, torch.fx.Node)
            if (match := self.match_feed_main_source(arg, output_meta)) is not None
        ]
        return self.validate_feed_main_source_match(
            source,
            GemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def feed_main_plan(
        self,
        output: torch.fx.Node,
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Find the feed-main match for an output: a view node searches its source; other
        # shape-preserving pointwise nodes recurse into node inputs and merge matches into one
        # physical value (validated). Returns None if not applicable.
        """Match feed-main reductions through trailing pointwise nodes."""
        normalized = self.graph.normalized_nodes.get(output)
        if isinstance(normalized, NormalizedView):
            return self.match_feed_main_source(
                normalized.source, output.meta.get("val")
            )
        if not is_shape_preserving_pointwise_node(output):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((output.args, output.kwargs))
            if isinstance(arg, torch.fx.Node)
            if (match := self.feed_main_plan(arg)) is not None
        ]
        return self.validate_feed_main_source_match(
            output,
            GemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def common_feed_main_match(
        self,
        candidates: tuple[Any, ...],
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Compute the feed-main plan of each node candidate and require all matches to
        # share one physical value; raises NotImplementedError (mixed feed-main error) otherwise.
        """Find the physical reduction value shared by feed-main consumers."""
        matches = [
            match
            for candidate in candidates
            if isinstance(candidate, torch.fx.Node)
            if (match := self.feed_main_plan(candidate)) is not None
        ]
        return GemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR
        )

    def common_reduction_dependency_match(
        self, outputs: Sequence[torch.fx.Node]
    ) -> GemmLocalReduceMatch | None:
        # [FUNCTION DOC] Collect outputs and their transitive dependencies; if every recorded match among
        # them refers to the same reduction node return the first, else None (also None if there
        # are no matches).
        """Return one physical reduction transitively consumed by the outputs."""
        reachable = OrderedSet(outputs)
        for output in outputs:
            reachable.update(self.graph.dependencies.get(output, ()))
        matches = [self.matches[node] for node in reachable if node in self.matches]
        if not matches:
            return None
        reduction_node = matches[0].reduction_node
        if reduction_node is None or any(
            match.reduction_node is not reduction_node for match in matches
        ):
            return None
        return matches[0]

    def compressed_aux_plan(
        self,
        output: Any,
        aux: torch.fx.Node,
        aux_index: int,
    ) -> GemmOutputLocalReducePlan | None:
        # [FUNCTION DOC] If aux has a match and its shape is statically known equal to the compressed
        # shape of the main output for that geometry, return a store-only plan at aux_index;
        # None when metadata is missing or the shape differs.
        """Plan a matched local reduction returned in compressed output shape."""
        match = self.matches.get(aux)
        output_meta = (
            output.meta.get("val") if isinstance(output, torch.fx.Node) else None
        )
        aux_meta = aux.meta.get("val")
        if match is None or aux_meta is None or output_meta is None:
            return None
        expected_aux_shape = local_reduce_compressed_shape(
            output_meta.shape, match.geometry.group, match.geometry.axis
        )
        if not statically_known_shape_equal(expected_aux_shape, aux_meta.shape):
            return None
        return match.to_plan(
            store=GemmLocalReduceStore(aux, aux_index), feeds_main=False
        )

    def feed_main_output_plan(
        self,
        output: torch.fx.Node,
        aux_outputs: tuple[torch.fx.Node, ...] = (),
        *,
        allow_dependency_match: bool = False,
    ) -> GemmOutputPlan | None:
        # [FUNCTION DOC] Build an output plan where one physical reduction value feeds the main output:
        # use the common feed-main match over output and aux outputs, optionally falling back to
        # the dependency match. Returns None if nothing matches. May raise NotImplementedError.
        """Plan one physical reduction value consumed by the main output."""
        match = self.common_feed_main_match((output, *aux_outputs))
        if match is None and allow_dependency_match:
            match = self.common_reduction_dependency_match((output, *aux_outputs))
        if match is None:
            return None
        return GemmOutputPlan(
            output,
            aux_outputs,
            match.to_plan(store=None, feeds_main=True),
        )
