# mypy: allow-untyped-defs
# [MODULE DOC] Partitioned scatter optimization pass for high-contention index_put
# (accumulate) ops on accelerators. Instead of many writes atomically accumulating
# into the same slots, writes are spread over a power-of-two number of partitions in
# an expanded buffer, the partitions are summed, and the result is added to the
# original input. Main contents: ScatterCandidate, ScatterMemoryState and
# ScatterPassContext (state), cheap pre-scan gates, a memory-budget estimate based
# on the memory profile, a pattern-matcher replacement, and the entry point
# partitioned_scatter_optimization_pass (the only exported name). Import-time side
# effects: creates module loggers and a partitioned_scatter artifact logger; no
# registration happens until the pass runs.
"""
Partitioned scatter optimization for high-contention index_put operations.

Algorithm:
  1. Assign each write operation a partition: partition_id = op_id & (P - 1)
  2. Scatter into an expanded buffer of size P * dim_size along scatter_dim
  3. Reshape to [..., P, dim_size, ...] and sum across partitions
  4. Add result to the original input
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes.memory_estimator import build_memory_profile
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._logging import getArtifactLogger
from torch.fx.experimental.symbolic_shapes import optimization_hint


log = logging.getLogger(__name__)
_artifact_log = getArtifactLogger(__name__, "partitioned_scatter")
aten = torch.ops.aten
prims = torch.ops.prims

_INDEX_PUT_TARGETS = (aten.index_put.default, aten.index_put_.default)


@dataclass
class ScatterCandidate:
    # [CLASS DOC] Immutable-by-convention record of an index_put op
    # (accumulate=True) that passed the cheap gates: output node, index node,
    # scatter dimension, element counts of output and index, size of the scatter
    # dimension, element size in bytes, contention ratio (index size divided by
    # scatter dim size) and dtype.
    """An index_put(accumulate=True) op that passed the cheap (non-memory) gates."""

    output_node: fx.Node
    index_node: fx.Node
    scatter_dim: int
    output_size: int
    index_size: int
    scatter_dim_size: int
    element_bytes: int
    contention_ratio: float
    dtype: torch.dtype


@dataclass
class ScatterMemoryState:
    # [CLASS DOC] Memory budget state for one pass invocation: peak live GPU bytes
    # per compute node (profile of the original graph at allocation time), mapping
    # from node to profile index, allowed peak (total GPU memory minus the non-model
    # floor), the raw totals, and committed_overhead_bytes, a running sum of
    # expanded-buffer bytes already granted to earlier transformed scatters (so
    # candidates do not all claim the same budget).
    # Peak live GPU bytes at each compute node in the original (un-transformed)
    # graph, taken at the allocation phase (before this node's last-use inputs
    # are freed) so the expanded scatter buffer is charged against the true peak.
    peak_mem_by_node: list[int]

    node_to_idx: dict[fx.Node, int]

    # total_gpu_memory - non_model_floor_bytes
    allowed_peak_bytes: int

    total_gpu_bytes: int
    non_model_floor_bytes: int

    # Expanded-buffer bytes already granted to scatters transformed earlier in this
    # invocation. peak_mem_by_node profiles the *original* graph, so without this every
    # candidate is sized against the same baseline and each claims the whole budget.
    # A running sum rather than a per-live-range charge because scatters sharing an
    # input fuse into one kernel, which needs all their expanded buffers live at once.
    committed_overhead_bytes: int = 0


@dataclass
class ScatterPassContext:
    # [CLASS DOC] Per-invocation mutable state shared with pattern callbacks via
    # closures: candidate map keyed by output node, lazily built memory state (None
    # when unavailable), counters of candidates and applied rewrites, per-reason
    # skip counts, and the list of partition counts applied.
    """Per-invocation state, passed to the pattern callbacks via closures."""

    # index_put nodes that survived the cheap pre-scan, keyed by output node.
    candidates: dict[fx.Node, ScatterCandidate] = field(default_factory=dict)

    # Built lazily, only when at least one candidate survives the pre-scan.
    memory: "ScatterMemoryState | None" = None

    n_candidates: int = 0
    n_applied: int = 0
    skip_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    applied_partitions: list[int] = field(default_factory=list)


def _record_skip(
    ctx: ScatterPassContext, reason: str, node_name: str, *args: Any
) -> None:
    # [FUNCTION DOC] HELPER: Record why a candidate was skipped: increments the
    # reason count in ctx and the inductor counter for that reason, and at debug
    # level logs the node name and any extra detail arguments to the artifact
    # logger.
    ctx.skip_reasons[reason] += 1
    counters["inductor"][f"partitioned_scatter_skipped_{reason}"] += 1
    if _artifact_log.isEnabledFor(logging.DEBUG):
        fmt = f"partitioned_scatter: SKIP node=%s reason={reason}"
        if args:
            fmt += " " + " ".join(str(a) for a in args)
        _artifact_log.debug(fmt, node_name)


def _evaluate_candidate(
    output_node: fx.Node, force: bool, ctx: ScatterPassContext
) -> "ScatterCandidate | None":
    # [FUNCTION DOC] Apply the cheap (non-memory) gates to an index_put node and
    # return a ScatterCandidate or None (recording a skip reason). Gates in order:
    # input is an FX node; exactly one non-None index; tensor metadata available;
    # device is not CPU; neither input nor index dtype is bool; scatter dim in
    # range; sizes resolvable and non-zero; index size at least the configured
    # minimum; contention ratio at least the configured threshold. The last two are
    # bypassed when force is true.
    # KNOWN ISSUE: the gate list says the op must run on a CUDA device, but the code
    # only rejects CPU, so any non-CPU device passes.
    """
    Cheap (non-memory) gates, in order:
      1. Single non-None index (multi-axis not supported)
      2. Valid tensor metadata
      3. Runs on a CUDA device (this optimization targets GPU atomic contention)
      4. Non-bool dtype
      5. scatter_dim in bounds
      6. Resolvable, non-zero sizes
      7. index_size >= min_index_size
      8. contention_ratio >= threshold  (uses scatter_dim_size, not output_numel)
    """
    node_name = output_node.name

    input_node = output_node.args[0]
    indices_arg = output_node.args[1]

    if not isinstance(input_node, fx.Node):
        _record_skip(ctx, "input_not_node", node_name)
        return None

    scatter_dim, index_node = _extract_scatter_dim_and_index(indices_arg)
    if scatter_dim is None or index_node is None:
        _record_skip(ctx, "multi_index", node_name)
        return None

    input_meta = _get_tensor_meta(input_node)
    index_meta = _get_tensor_meta(index_node)
    if not input_meta or not index_meta:
        _record_skip(ctx, "no_meta", node_name)
        return None

    # Only rewrite accelerator ops: the expanded-buffer trade-off targets GPU
    # atomic contention and is a pessimization on CPU. Matches the non-CPU
    # device_filter that the memory estimator uses.
    if input_meta["device"].type == "cpu":
        _record_skip(ctx, "cpu_device", node_name)
        return None

    if input_meta["dtype"] == torch.bool or index_meta["dtype"] == torch.bool:
        _record_skip(ctx, "bool_dtype", node_name)
        return None

    if scatter_dim >= len(input_meta["shape"]):
        _record_skip(ctx, "dim_out_of_bounds", node_name)
        return None

    output_size = _resolve_numel(input_meta["numel"])
    index_size = _resolve_numel(index_meta["numel"])

    if output_size is None or index_size is None:
        _record_skip(ctx, "dynamic_no_hint", node_name)
        return None

    if output_size == 0 or index_size == 0:
        _record_skip(ctx, "zero_size", node_name)
        return None

    # Contention is per scatter-dim slot, so use scatter_dim_size as denominator.
    # For a [vocab, dim] output, ratio = N/vocab, not N/(vocab*dim).
    scatter_dim_size = _resolve_numel(input_meta["shape"][scatter_dim])
    if scatter_dim_size is None or scatter_dim_size == 0:
        _record_skip(ctx, "zero_size", node_name)
        return None

    min_index_size: int = config.partitioned_scatter_min_index_size
    if not force and index_size < min_index_size:
        _record_skip(
            ctx,
            "index_too_small",
            node_name,
            f"index_size={index_size} min={min_index_size}",
        )
        return None

    contention_ratio = index_size / scatter_dim_size
    min_contention: float = config.partitioned_scatter_min_contention_ratio
    if not force and contention_ratio < min_contention:
        _record_skip(
            ctx,
            "low_contention",
            node_name,
            f"contention_ratio={contention_ratio:.3f} threshold={min_contention:.3f}",
        )
        return None

    return ScatterCandidate(
        output_node=output_node,
        index_node=index_node,
        scatter_dim=scatter_dim,
        output_size=output_size,
        index_size=index_size,
        scatter_dim_size=scatter_dim_size,
        element_bytes=input_meta["dtype"].itemsize,
        contention_ratio=contention_ratio,
        dtype=input_meta["dtype"],
    )


def _scan_candidates(graph: fx.Graph, ctx: ScatterPassContext) -> None:
    # [FUNCTION DOC] HELPER: Pre-scan the graph for index_put or in-place index_put
    # calls whose fourth positional argument is the literal true (accumulate), count
    # them in ctx.n_candidates, and store those that pass the cheap gates in
    # ctx.candidates. Lets the pass avoid building the memory profile when nothing
    # qualifies. Calls with accumulate passed by keyword are not counted.
    """
    Cheap pre-scan for index_put(accumulate=True) ops that could be rewritten.

    Applies every gate that does not require the memory profile (device, dtype,
    shape, min index size, contention ratio) so we can skip building the
    memory profile entirely when no op would qualify.
    """
    force: bool = config.partitioned_scatter_force

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in _INDEX_PUT_TARGETS:
            continue
        args = node.args
        if len(args) < 4 or args[3] is not True:
            continue

        ctx.n_candidates += 1
        candidate = _evaluate_candidate(node, force, ctx)
        if candidate is not None:
            ctx.candidates[node] = candidate


def _build_scatter_memory_state(graph: fx.Graph) -> "ScatterMemoryState | None":
    # [FUNCTION DOC] HELPER: Build the memory state from a per-node peak memory
    # profile of the original graph. Returns None if CUDA is unavailable or the
    # profile cannot be built (any exception is logged at debug and swallowed), in
    # which case the pass runs without memory gating. Uses total GPU memory minus
    # the configured non-model floor as the allowed peak, and indexes the
    # allocation-time entry (every second profile entry) of each compute node.
    """
    Build a per-node peak-memory profile of the original graph.
    Returns None when CUDA is unavailable or the profile can't be built; the
    pass then runs unconstrained by the memory budget.
    """
    if not torch.cuda.is_available():
        return None

    _, total_gpu = torch.cuda.mem_get_info()

    floor_bytes: int = config.partitioned_scatter_non_model_floor_bytes
    allowed_peak = max(0, total_gpu - floor_bytes)

    def is_releasable(n: fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Predicate for the memory profile: a node's memory
        # is releasable unless its name starts with primals.
        return not n.name.startswith("primals")

    # build_memory_profile emits two entries per compute node: the first after
    # its allocations (the node-local peak, while last-use inputs are still
    # live) and the second after its deallocations. We key on the allocation
    # entry so the expanded scatter buffer is charged against the real peak.
    # Unbacked symbolic sizes have no optimization_hint, so sizing the profile
    # can raise. Memory gating is a heuristic, so fall back to running the pass
    # unconstrained rather than failing the compile.
    try:
        profile = build_memory_profile(graph, is_releasable)
    except Exception as e:
        _artifact_log.debug(
            "partitioned_scatter: build_memory_profile failed (%s), "
            "running without memory gating",
            e,
        )
        return None

    compute_nodes = [
        n for n in graph.nodes if n.op not in ("placeholder", "get_attr", "output")
    ]
    node_to_idx: dict[fx.Node, int] = {}
    peak_mem: list[int] = []
    for i, node in enumerate(compute_nodes):
        alloc_idx = 1 + 2 * i
        if alloc_idx >= len(profile):
            break
        node_to_idx[node] = i
        peak_mem.append(profile[alloc_idx])

    _artifact_log.debug(
        "partitioned_scatter: memory state built — "
        "graph_nodes=%d compute_nodes=%d "
        "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
        sum(1 for _ in graph.nodes),
        len(compute_nodes),
        total_gpu // 1_000_000,
        floor_bytes // 1_000_000,
        allowed_peak // 1_000_000,
    )

    return ScatterMemoryState(
        peak_mem_by_node=peak_mem,
        node_to_idx=node_to_idx,
        allowed_peak_bytes=allowed_peak,
        total_gpu_bytes=total_gpu,
        non_model_floor_bytes=floor_bytes,
    )


def _compute_num_partitions(
    available_bytes: int,
    output_size: int,
    element_bytes: int,
    min_p: int,
    max_p: int,
    index_size: int = 0,
    scatter_dim_size: int = 0,
    force: bool = False,
) -> int:
    # [FUNCTION DOC] HELPER: Return the largest power-of-two partition count P in
    # the range min_p to max_p such that the extra memory output_size times
    # element_bytes times (P minus 1) fits in available_bytes, additionally capped
    # (unless force) at about 4 times the writes per scatter-dim slot but never
    # below min_p. Returns 0 if available bytes, output size or element size is non-
    # positive, or min_p does not fit. Power of two is needed for the bitwise-and
    # partition assignment.
    """
    Return the largest power-of-2 P in [min_p, max_p] satisfying:
      1. Memory: output_size * element_bytes * (P - 1) <= available_bytes
      2. Diminishing-returns cap (skipped when force=True):
         P <= 4 * writes_per_slot, where writes_per_slot = index_size / scatter_dim_size.
         Past ~4W partitions for a slot taking W writes most partition slots are
         never written, so the zero-fill and reduce over them buy nothing.

    Returns 0 if min_p doesn't fit. Power-of-2 is required by the bitwise-AND
    partition assignment.
    """
    if available_bytes <= 0 or output_size == 0 or element_bytes == 0:
        return 0

    max_raw = available_bytes / (output_size * element_bytes) + 1
    if max_raw < min_p:
        return 0

    p = 2 ** int(math.log2(max_raw))
    p = min(p, max_p)

    if not force and index_size > 0 and scatter_dim_size > 0:
        writes_per_slot = index_size / scatter_dim_size
        contention_cap = max(min_p, 2 ** int(math.log2(max(1, 4 * writes_per_slot))))
        p = min(p, contention_cap)

    return p


def _check_memory(
    state: ScatterMemoryState,
    candidate: ScatterCandidate,
    force: bool = False,
) -> int:
    # [FUNCTION DOC] HELPER: Compute the partition count for a candidate from the
    # memory state: available bytes are the allowed peak minus the node's baseline
    # peak minus committed overhead. Returns 0 (fail closed) when the node has no
    # profile entry; otherwise the result of the partition computation. Logs details
    # at debug level.
    """
    Compute num_partitions for this candidate given the peak-memory profile.
    Returns num_partitions >= min_p, or 0 if the memory constraint cannot be met.
    """
    min_p = config.partitioned_scatter_min_partitions
    max_p = config.partitioned_scatter_max_partitions

    idx = state.node_to_idx.get(candidate.output_node)
    if idx is None:
        # No profile entry means we cannot bound this scatter's peak, so fail
        # closed: granting max_p here would allocate max_p-1 extra copies of the
        # output with no memory check at all.
        return 0

    baseline = state.peak_mem_by_node[idx]
    available = state.allowed_peak_bytes - baseline - state.committed_overhead_bytes

    num_partitions = _compute_num_partitions(
        available,
        candidate.output_size,
        candidate.element_bytes,
        min_p,
        max_p,
        index_size=candidate.index_size,
        scatter_dim_size=candidate.scatter_dim_size,
        force=force,
    )

    if _artifact_log.isEnabledFor(logging.DEBUG):
        overhead = (
            candidate.output_size * candidate.element_bytes * max(0, num_partitions - 1)
        )
        _artifact_log.debug(
            "partitioned_scatter: memory check node=%s "
            "baseline_peak=%d MB committed=%d MB available=%d MB "
            "expanded_buffer_cost=%d MB num_partitions=%d "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            candidate.output_node.name,
            baseline // 1_000_000,
            state.committed_overhead_bytes // 1_000_000,
            available // 1_000_000,
            overhead // 1_000_000,
            num_partitions,
            state.total_gpu_bytes // 1_000_000,
            state.non_model_floor_bytes // 1_000_000,
            state.allowed_peak_bytes // 1_000_000,
        )

    return num_partitions


def _resolve_numel(numel: Any) -> int | None:
    # [FUNCTION DOC] HELPER: Convert a numel or size to a concrete integer. A SymInt
    # is resolved through its optimization hint and doubled as a safety margin;
    # returns None when no hint exists. Other values are converted to int.
    """Resolve numel to a concrete int, handling SymInt via optimization_hint."""
    if isinstance(numel, torch.SymInt):
        hint = optimization_hint(numel)
        if hint is None:
            return None
        return hint * 2  # 2× safety margin for dynamic shapes
    return int(numel)


def _validate_memory(match: Match, ctx: ScatterPassContext, force: bool) -> bool:
    # [FUNCTION DOC] HELPER: Second-stage gate of the pattern matcher (extra check):
    # look up the candidate for the matched output node, compute the partition count
    # with or without the memory state, and reject (recording a memory_budget skip)
    # if it is below the configured minimum. On success stores num_partitions,
    # scatter dim, index node, output size and element bytes as attributes on the
    # match object for the replacement, and returns True. Returns False if the node
    # was not a pre-scan candidate.
    """
    Second-stage gate (the memory budget) for a matched index_put node.

    The cheap gates already ran in the pre-scan; here we only look up the
    surviving candidate and size the partition count against the memory budget.
    """
    output_node = match.output_node()
    candidate = ctx.candidates.get(output_node)
    if candidate is None:
        # Node failed a cheap gate in the pre-scan (already recorded).
        return False

    min_p: int = config.partitioned_scatter_min_partitions
    max_p: int = config.partitioned_scatter_max_partitions

    if ctx.memory is not None:
        num_partitions = _check_memory(ctx.memory, candidate, force=force)
    else:
        num_partitions = _compute_num_partitions(
            available_bytes=2**62,
            output_size=candidate.output_size,
            element_bytes=candidate.element_bytes,
            min_p=min_p,
            max_p=max_p,
            index_size=candidate.index_size,
            scatter_dim_size=candidate.scatter_dim_size,
            force=force,
        )

    if num_partitions < min_p:
        _record_skip(
            ctx,
            "memory_budget",
            output_node.name,
            f"num_partitions={num_partitions} min={min_p}",
        )
        return False

    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = candidate.scatter_dim  # type: ignore[attr-defined]
    match._index_node = candidate.index_node  # type: ignore[attr-defined]
    match._output_size = candidate.output_size  # type: ignore[attr-defined]
    match._element_bytes = candidate.element_bytes  # type: ignore[attr-defined]

    if _artifact_log.isEnabledFor(logging.DEBUG):
        _artifact_log.debug(
            "partitioned_scatter: APPLY node=%s "
            "num_partitions=%d scatter_dim=%d "
            "contention_ratio=%.1f (index_size=%d / scatter_dim_size=%d) "
            "output_size=%d dtype=%s%s",
            output_node.name,
            num_partitions,
            candidate.scatter_dim,
            candidate.contention_ratio,
            candidate.index_size,
            candidate.scatter_dim_size,
            candidate.output_size,
            candidate.dtype,
            " [force]" if force else "",
        )

    return True


def _create_replacement(
    match: Match, ctx: ScatterPassContext, input_tensor, indices, values
) -> None:
    # [FUNCTION DOC] HELPER: Pattern replacement callback: rewrites the matched
    # index_put by tracing a replacement function with the matcher
    # (replace_by_example) using the stored partition count, scatter dim and index
    # node, then charges the expanded buffer bytes to the memory state, increments
    # the applied counters, records the partition count and bumps the inductor
    # counter.
    """Replace high-contention index_put with partitioned scatter."""
    num_partitions: int = match._num_partitions  # type: ignore[attr-defined]
    scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
    index_node = match._index_node  # type: ignore[attr-defined]

    def repl(input_tensor, index_node, values):
        # [FUNCTION DOC] HELPER: Traced replacement body: flatten multi-dimensional
        # indices and corresponding values to 1-D; assign write i to partition i
        # bitwise-and (num_partitions minus 1); allocate a zero buffer whose scatter
        # dimension is multiplied by num_partitions; shift each index by partition
        # id times original dim size; accumulate-scatter into the buffer; view as
        # partitions by dim size along the scatter dim and sum over the partition
        # axis (keeping small integer dtypes); finally return the input plus the
        # reduced result.
        dim_size = input_tensor.shape[scatter_dim]
        num_operations = index_node.numel()

        # Flatten multi-dimensional indices to 1-D
        if len(index_node.shape) > 1:
            flat_index = index_node.reshape(num_operations)
            values_ndim = len(index_node.shape)
            flat_values = values.reshape(
                [num_operations] + list(values.shape[values_ndim:])
            )
        else:
            flat_index = index_node
            flat_values = values

        # partition_id = op_id & (num_partitions - 1), requires power-of-2
        operation_ids = torch.ops.prims.iota.default(
            num_operations,
            start=0,
            step=1,
            dtype=flat_index.dtype,
            device=flat_index.device,
            requires_grad=False,
        )
        partition_ids = torch.ops.aten.bitwise_and.Scalar(
            operation_ids, num_partitions - 1
        )

        # Expanded buffer: one copy per partition along scatter_dim
        expanded_shape = list(input_tensor.shape)
        expanded_shape[scatter_dim] *= num_partitions
        expanded_buffer = torch.ops.aten.full.default(
            expanded_shape,
            0,
            dtype=flat_values.dtype,
            layout=torch.strided,
            device=flat_values.device,
            pin_memory=False,
        )

        # Shift each write into its partition's slice
        partition_offsets = partition_ids * dim_size
        adjusted_index = flat_index + partition_offsets

        if isinstance(indices, (list, tuple)):
            adjusted_indices = [
                adjusted_index if i == scatter_dim else idx
                for i, idx in enumerate(indices)
            ]
        else:
            adjusted_indices = [adjusted_index]

        scattered_buffer = torch.ops.aten.index_put.default(
            expanded_buffer, adjusted_indices, flat_values, True
        )

        # Reshape to [..., num_partitions, dim_size, ...] then sum partitions
        reduce_shape = list(expanded_shape)
        reduce_shape[scatter_dim] = num_partitions
        reduce_shape.insert(scatter_dim + 1, dim_size)
        reshaped = torch.ops.aten.view.default(scattered_buffer, reduce_shape)

        # Preserve dtype for integer types that don't promote during sum
        if flat_values.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            reduced = torch.ops.aten.sum.dim_IntList(
                reshaped, [scatter_dim], dtype=flat_values.dtype
            )
        else:
            reduced = torch.ops.aten.sum.dim_IntList(reshaped, [scatter_dim])

        return input_tensor + reduced

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [input_tensor, index_node, values])

    # Charge this scatter's expanded buffer so later candidates in the same
    # invocation see a correspondingly smaller budget.
    if ctx.memory is not None:
        output_size: int = match._output_size  # type: ignore[attr-defined]
        element_bytes: int = match._element_bytes  # type: ignore[attr-defined]
        ctx.memory.committed_overhead_bytes += (
            output_size * element_bytes * (num_partitions - 1)
        )

    ctx.n_applied += 1
    ctx.applied_partitions.append(num_partitions)
    counters["inductor"]["partitioned_scatter_applied"] += 1


def _build_pattern_pass(ctx: ScatterPassContext) -> PatternMatcherPass:
    # [FUNCTION DOC] HELPER: Create a fresh PatternMatcherPass for this invocation
    # and register the index_put (and in-place) accumulate patterns with callbacks
    # bound to ctx, so no module-level mutable state is shared across concurrent
    # compilations. Returns the pass.
    """
    Construct a per-invocation PatternMatcherPass whose callbacks close over
    `ctx`, avoiding module-level global state (which is unsafe under concurrent
    compilation).
    """
    patterns = PatternMatcherPass(pass_name="partitioned_scatter_optimization")
    force: bool = config.partitioned_scatter_force

    def extra_check(match: Match) -> bool:
        # [FUNCTION DOC] HELPER: Pattern extra check: delegate to the memory
        # validation with the captured context and force flag.
        return _validate_memory(match, ctx, force)

    def replacement(match: Match, input_tensor, indices, values) -> None:
        # [FUNCTION DOC] HELPER: Pattern replacement: delegate to the replacement
        # builder with the captured context.
        _create_replacement(match, ctx, input_tensor, indices, values)

    for target in _INDEX_PUT_TARGETS:
        register_graph_pattern(
            CallFunction(target, Arg(), Arg(), Arg(), True),
            extra_check=extra_check,
            pass_dict=patterns,  # type: ignore[arg-type]
        )(replacement)

    return patterns


def _log_summary(ctx: ScatterPassContext, num_matches: int) -> None:
    # [FUNCTION DOC] HELPER: Log an info-level summary (candidates, applied,
    # skipped, partitions per op, skip breakdown and memory numbers if available)
    # when at least one candidate existed. The num_matches parameter is unused.
    if ctx.n_candidates == 0:
        return

    if ctx.memory is not None:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s "
            "skip_breakdown=%s "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            ctx.n_candidates,
            ctx.n_applied,
            ctx.n_candidates - ctx.n_applied,
            ctx.applied_partitions,
            dict(ctx.skip_reasons),
            ctx.memory.total_gpu_bytes // 1_000_000,
            ctx.memory.non_model_floor_bytes // 1_000_000,
            ctx.memory.allowed_peak_bytes // 1_000_000,
        )
    else:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s skip_breakdown=%s (no memory state)",
            ctx.n_candidates,
            ctx.n_applied,
            ctx.n_candidates - ctx.n_applied,
            ctx.applied_partitions,
            dict(ctx.skip_reasons),
        )


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    # [FUNCTION DOC] Entry point: apply the partitioned scatter optimization to the
    # graph when config.partitioned_scatter_enabled is set. Stage one pre-scans for
    # candidates; if none, returns early. Stage two builds the memory state, runs
    # the pattern pass, logs a summary and lints the graph if any rewrites happened.
    # Mutates and returns the same graph.
    """
    Apply partitioned scatter optimization to high-contention index_put operations.
    Controlled by config.partitioned_scatter_enabled.
    """
    if not config.partitioned_scatter_enabled:
        return graph

    ctx = ScatterPassContext()

    # Stage 1: cheap pre-scan. If nothing qualifies we avoid building the
    # (relatively expensive) memory profile entirely.
    _scan_candidates(graph, ctx)
    if not ctx.candidates:
        _log_summary(ctx, 0)
        return graph

    # Stage 2: build the memory profile and run the pattern matcher.
    ctx.memory = _build_scatter_memory_state(graph)
    patterns = _build_pattern_pass(ctx)
    num_matches = patterns.apply(graph)

    _log_summary(ctx, num_matches)

    if num_matches > 0:
        graph.lint()

    return graph


def _extract_scatter_dim_and_index(
    indices_arg: Any,
) -> tuple[int | None, fx.Node | None]:
    # [FUNCTION DOC] HELPER: Return the scatter dimension and index node from an
    # index_put indices argument. A non-sequence argument is treated as the index
    # for dimension 0. For a sequence, returns the position and value of the single
    # non-None entry, or a pair of None values if there are several (or none).
    """Extract scatter dimension and index node from indices argument."""
    if not isinstance(indices_arg, (list, tuple)):
        return 0, indices_arg

    index_node = None
    scatter_dim = None

    for dim, idx in enumerate(indices_arg):
        if idx is not None:
            if index_node is not None:
                return None, None
            index_node = idx
            scatter_dim = dim

    return scatter_dim, index_node


def _get_tensor_meta(node: fx.Node) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: Return a mapping with shape (tuple), dtype, device and
    # numel taken from the node's fake value, or None if the node has no value or
    # the value lacks shape or dtype.
    """Extract tensor metadata from an FX node's meta['val'] FakeTensor."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None

    val = node.meta["val"]
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
        return None

    return {
        "shape": tuple(val.shape),
        "dtype": val.dtype,
        "device": val.device,
        "numel": val.numel(),
    }


__all__ = ["partitioned_scatter_optimization_pass"]
