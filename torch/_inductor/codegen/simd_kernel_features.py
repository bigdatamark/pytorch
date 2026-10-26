# [MODULE DOC] Feature extraction for SIMD (Triton-style) kernels in Inductor codegen.
# Defines node-schedule markers (DisableReduction/EnableReduction), SIMDKernelFeatures
# (an ordered node schedule plus numel/reduction_numel with cached queries such as index
# dtype selection, reduction hints, op counts), and a memory-traffic estimator
# (MemoryEstimator -> MemoryStats) used by tiling/config heuristics.
# No import-time side effects beyond defining constants and classes.
from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import typing
from typing import Any

import sympy

import torch

from ...utils._ordered_set import OrderedSet
from ...utils._sympy.functions import FloorDiv, Min, ModularIndexing
from ...utils._sympy.symbol import make_symbol, SymT
from .. import ir
from ..dependencies import Dep, extract_loop_body_with_args, MemoryDep
from ..runtime.hints import ReductionHint
from ..runtime.runtime_utils import next_power_of_2
from ..scheduler import SchedulerNode
from ..utils import cache_on_self
from ..virtualized import V


if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch._inductor.tiling_utils import CoalesceVarAnalysis


_INNER_REDUCTION_RATIO = 32
_SMALL_INNER_REDUCTION_RATIO = 16
_SMALL_INNER_REDUCTION_MAX_RBLOCK = 512


def tiling_scores_suggest_inner_reduction(
    tiling_scores: dict[str, sympy.Expr], reduction_numel: sympy.Expr
) -> bool:
    # [FUNCTION DOC] Decide whether a reduction should be treated as an inner (contiguous) reduction
    # from tiling scores. Sums the x/y/z scores (floored at 1) as the non-reduction score and
    # compares with the "r0_" score using optimization hints: a ratio at or above 32 returns True,
    # below 16 returns False; in between returns True only if the reduction numel rounded up to a
    # power of two is at most 512. Args: tiling_scores (dim name -> score), reduction_numel.
    # Returns bool. Raises KeyError if "r0_" is missing. Reads V.graph.sizevars.
    """Return whether tiling scores justify treating a reduction as inner."""
    sizevars = V.graph.sizevars
    non_reduction_score = sum(
        sizevars.optimization_hint(tiling_scores.get(dim, 0), fallback=0)
        for dim in ("x", "y", "z")
    )
    non_reduction_score = max(non_reduction_score, 1)
    r_score = sizevars.optimization_hint(tiling_scores["r0_"], fallback=0)
    if r_score >= _INNER_REDUCTION_RATIO * non_reduction_score:
        return True
    if r_score < _SMALL_INNER_REDUCTION_RATIO * non_reduction_score:
        return False

    # Moderate score ratios are useful while persistent configs can keep XBLOCK=8.
    rblock_hint = next_power_of_2(
        max(sizevars.optimization_hint(reduction_numel, fallback=1), 1)
    )
    return rblock_hint <= _SMALL_INNER_REDUCTION_MAX_RBLOCK


class NodeScheduleMarker:
    # [CLASS DOC] Base class for marker entries placed in a node schedule alongside
    # SchedulerNodes. Markers are used as classes (never instantiated); they are sentinels
    # that change reduction state during codegen. Has no state.
    @staticmethod
    def only_nodes(it: Iterable[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        # [FUNCTION DOC] Generator that yields each entry of the schedule which is not one of the
        # DisableReduction/EnableReduction markers, i.e. only the real scheduler nodes, in order.
        for item in it:
            if not (item is DisableReduction or item is EnableReduction):
                yield item  # type: ignore[misc]

    @staticmethod
    def is_reduction() -> bool:
        # [FUNCTION DOC] Markers are never reductions; always returns False. Lets markers be
        # duck-typed like nodes.
        return False


NodeScheduleEntry = SchedulerNode | type[NodeScheduleMarker]


class DisableReduction(NodeScheduleMarker):
    # [CLASS DOC] Sentinel marker: nodes after it (until EnableReduction) run outside the
    # reduction loop. Stateless; used by identity comparison on the class itself.
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction(NodeScheduleMarker):
    # [CLASS DOC] Sentinel marker ending a DisableReduction block, re-entering the reduction
    # loop. Stateless; provides the filter helper over node schedules.
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule: list[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        # [FUNCTION DOC] Generator over a node schedule yielding only nodes that are not inside a
        # DisableReduction..EnableReduction span. Markers themselves are never yielded.
        # Args: node_schedule (list of nodes and markers).
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node  # type: ignore[misc]


class SIMDKernelFeatures:
    # [CLASS DOC] Immutable-ish description of one kernel to generate: the node_schedule,
    # numel (non-reduction elements, simplified), reduction_numel (1 means no reduction),
    # optional coalesce_analysis and tiling_scores. Holds a per-instance cache of memory stats
    # keyed by group sizes. Many queries are memoized with cache_on_self.
    """
    An ordered schedule of nodes that will become a single kernel.
    """

    def __init__(
        self,
        node_schedule: list[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr = sympy.S.One,
        coalesce_analysis: CoalesceVarAnalysis | None = None,
        tiling_scores: dict[str, sympy.Expr] | None = None,
    ):
        # [FUNCTION DOC] Store the schedule, simplify numel and reduction_numel through the
        # graph sizevars, create an empty memory stats cache, and keep coalesce_analysis and
        # tiling_scores (both may be None). Requires an active V.graph.
        self.node_schedule = node_schedule
        # numel excludes reduction_numel
        self.numel: sympy.Expr = V.graph.sizevars.simplify(numel)
        self.reduction_numel: sympy.Expr = V.graph.sizevars.simplify(reduction_numel)
        self._stats_cache: dict[tuple[sympy.Expr, ...], MemoryStats] = {}
        self.coalesce_analysis = coalesce_analysis
        self.tiling_scores = tiling_scores

    def with_tiling_scores(
        self, tiling_scores: dict[str, sympy.Expr] | None
    ) -> SIMDKernelFeatures:
        # [FUNCTION DOC] Return a new SIMDKernelFeatures sharing the same schedule, sizes and
        # coalesce analysis but with the given tiling_scores (or None). Caches are not shared.
        return SIMDKernelFeatures(
            self.node_schedule,
            self.numel,
            self.reduction_numel,
            self.coalesce_analysis,
            tiling_scores,
        )

    @cache_on_self
    def is_reduction(self) -> bool:
        # [FUNCTION DOC] True when the kernel has a reduction, i.e. reduction_numel is not 1. Cached.
        return self.reduction_numel != 1

    @cache_on_self
    def scheduler_nodes(self) -> Iterable[SchedulerNode]:
        # [FUNCTION DOC] Return a tuple of the schedule's SchedulerNodes in order, with markers
        # removed. Cached.
        return tuple(NodeScheduleMarker.only_nodes(self.node_schedule))

    def reduction_nodes(self) -> list[SchedulerNode]:
        # [FUNCTION DOC] Return the list of scheduler nodes that are reductions, in schedule order.
        return [n for n in self.scheduler_nodes() if n.is_reduction()]

    @cache_on_self
    def strict_sum_reductions(self) -> tuple[ir.Reduction, ...]:
        # [FUNCTION DOC] Return a tuple of the Reduction IR objects (from ComputedBuffer reduction nodes)
        # whose strict_sum_rblock is set, i.e. reductions requiring strict-sum ordering. Cached.
        return tuple(
            node.node.data
            for node in self.reduction_nodes()
            if isinstance(node.node, ir.ComputedBuffer)
            and isinstance(node.node.data, ir.Reduction)
            and node.node.data.strict_sum_rblock is not None
        )

    def has_strict_sum_multirow_reduction(self) -> bool:
        # [FUNCTION DOC] True if any strict-sum reduction in the kernel is flagged multirow.
        return any(r.strict_sum_multirow for r in self.strict_sum_reductions())

    @cache_on_self
    def strict_sum_rblock(self) -> int | None:
        # [FUNCTION DOC] Return the single reduction block size shared by all strict-sum reductions,
        # or None when there are none. Raises AssertionError if they disagree on the size. Cached.
        rblocks = OrderedSet(
            reduction.strict_sum_rblock for reduction in self.strict_sum_reductions()
        )
        if not rblocks:
            return None
        if len(rblocks) != 1:
            raise AssertionError(
                f"strict sums require one reduction block size, got {rblocks}"
            )
        return next(iter(rblocks))

    @cache_on_self
    def buf_accesses(self) -> dict[str, list[Dep]]:
        # [FUNCTION DOC] Build a mapping from buffer name to the list of all read and write
        # dependencies touching it across the kernel's nodes. Cached.
        """only needed for config.benchmark_kernel"""
        buf_accesses = collections.defaultdict(list)
        for node in self.scheduler_nodes():
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)
        return buf_accesses

    @cache_on_self
    def op_counts(self) -> collections.Counter[str]:
        # [FUNCTION DOC] Aggregate the per-node loop body op counts into one Counter of op name to
        # number of occurrences across the kernel. Cached.
        counts: collections.Counter[str] = collections.Counter()
        for node in self.scheduler_nodes():
            counts.update(node._body.op_counts)
        return counts

    def contains_op(self, op_name: str) -> bool:
        # [FUNCTION DOC] True if the named op appears at least once in the kernel's op counts.
        """True if V.ops.{op_name} is used in node_schedule"""
        return bool(self.op_counts().get(op_name))

    def get_mutations(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Collect into an ordered set the names of buffers mutated by the outputs of
        # all scheduler nodes in the kernel.
        mutations: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            for buf in node.get_outputs():
                mutations.update(buf.get_mutations())
        return mutations

    @cache_on_self
    def select_index_dtype(self) -> torch.dtype:
        # [FUNCTION DOC] Choose the integer dtype for index math: int32 if the total numel
        # (numel times reduction_numel) and all buffers used by the nodes permit 32-bit indexing
        # (via SIMDScheduling.can_use_32bit_indexing) and no fused index expression can overflow
        # int32; otherwise int64. Looks up buffers in V.graph. Cached.
        # Gather all used buffer names
        buffer_names: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            buffer_names.update(node.get_buffer_names())
            buffer_names.update(node.used_buffer_names())
        buffers = [V.graph.get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = self.numel * self.reduction_numel

        from .simd import SIMDScheduling

        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            # Fused address expressions can overflow int32 even when
            # numel/storage_size are within range, so check them directly.
            if self.any_index_expr_overflows_int32():
                return torch.int64
            return torch.int32
        return torch.int64

    @cache_on_self
    def any_index_expr_overflows_int32(self) -> bool:
        # [FUNCTION DOC] Scan every MemoryDep read/write index in the kernel and return True if one
        # may exceed the 32-bit signed range, using the constant-offset check and the
        # variable-scaled upper-bound check described below. Arithmetic errors during
        # evaluation are swallowed and treated as "unknown" (not overflow). Cached.
        """Return True if any MemoryDep addressing expression can overflow int32.

        Two complementary checks are applied to each index:

        - Constant offset: the index evaluated at the origin (every loop
          variable set to 0) must lie within [-2**31, 2**31 - 1]. This is cheap
          and independent of loop-var ranges, so it also covers deps whose sizes
          are symbolic.
        - Variable-scaled bound: the index bounded over the actual iteration
          range of every loop variable must not exceed 2**31 - 1. This catches
          overflow such as the `V * x0` synthesized when unrolled chunked-loop
          slices are stitched into one pointwise kernel (`V` a per-chunk stride,
          `x0` spanning the fused numel); unlike a `numel * max_stride` proxy it
          is exact, since a stride on a size-1 dim multiplies an empty range and
          drops out. Deps with a symbolic loop-var size, or whose bound is
          unknown/indirect (infinite), are skipped here and left to the
          constant-offset and per-buffer storage-size checks.
        """
        from torch.utils._sympy.numbers import int_oo
        from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

        int32_max = sympy.Integer(2**31 - 1)
        int32_min = sympy.Integer(-(2**31))
        for node in self.scheduler_nodes():
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes):
                if not isinstance(dep, MemoryDep):
                    continue
                index = dep.index
                if not isinstance(index, sympy.Expr):
                    continue

                # Constant offset (index at the origin).
                try:
                    const_part = index.subs(
                        {s: sympy.Integer(0) for s in index.free_symbols}
                    )
                    if isinstance(const_part, sympy.Expr) and (
                        const_part > int32_max or const_part < int32_min
                    ):
                        return True
                except (ZeroDivisionError, TypeError, ValueError):
                    pass

                # Variable-scaled bound over concrete loop-var ranges. A
                # ValueRanges bound must be concrete, so skip a dep with any
                # symbolic loop-var size.
                sizes = dep.size
                if any(getattr(s, "free_symbols", None) for s in sizes):
                    continue
                try:
                    var_ranges = {
                        var: ValueRanges(0, max(int(size) - 1, 0))
                        for var, size in zip(dep.var_names, sizes)
                    }
                    upper = bound_sympy(index, var_ranges).upper
                    if (
                        isinstance(upper, sympy.Expr)
                        and not upper.has(int_oo, sympy.oo)
                        and upper > int32_max
                    ):
                        return True
                except (ZeroDivisionError, TypeError, ValueError):
                    continue
        return False

    def get_reduction_hint(
        self, tiling_scores: dict[str, sympy.Expr] | None = None
    ) -> ReductionHint:
        # [FUNCTION DOC] Compute the ReductionHint for the kernel. No reductions gives DEFAULT.
        # Otherwise use the common per-node hint (DEFAULT if they disagree); an INNER hint is
        # downgraded to DEFAULT when a non-contiguous pointwise node is in the kernel; a DEFAULT
        # hint is upgraded to INNER when tiling scores (argument, else self.tiling_scores)
        # contain both "x" and "r0_" and suggest an inner reduction.
        if tiling_scores is None:
            tiling_scores = self.tiling_scores
        reductions = self.reduction_nodes()
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel()
            ):
                reduction_hint_val = ReductionHint.DEFAULT

            # Upgrade DEFAULT to INNER for inner reductions based on tiling scores
            if (
                reduction_hint_val == ReductionHint.DEFAULT
                and tiling_scores is not None
                and "x" in tiling_scores
                and "r0_" in tiling_scores
            ):
                if tiling_scores_suggest_inner_reduction(
                    tiling_scores, self.reduction_numel
                ):
                    reduction_hint_val = ReductionHint.INNER
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        return reduction_hint_val

    @cache_on_self
    def buffer_read_counts(self) -> dict[str, int]:
        # [FUNCTION DOC] Return a plain mapping of buffer name to the number of read dependencies
        # on it across all scheduler nodes. Cached.
        """Counts how many times each buffer is read within the kernel"""
        read_counts: dict[str, int] = collections.defaultdict(int)

        for node in self.scheduler_nodes():
            # node.read_writes.reads contains MemoryDep objects for each read
            for read_dep in node.read_writes.reads:
                read_counts[read_dep.name] += 1

        return dict(read_counts)  # Convert defaultdict to regular dict

    def has_non_contiguous_pw_in_reduction_kernel(self) -> bool:
        # [FUNCTION DOC] True if some non-reduction node spanning the full numel (numel times
        # reduction_numel) has a memory dependency that is not contiguous, not an integer index
        # (e.g. random seed load), and does not have stride 1 on the last dim.
        pointwise_nodes = [
            n
            for n in self.scheduler_nodes()
            if not n.is_reduction()
            and n.group[1][0] == self.numel * self.reduction_numel
        ]
        for node in pointwise_nodes:
            # An index can be an integer when loading a random seed.
            if not all(
                not isinstance(dep, MemoryDep)
                or dep.is_contiguous()
                or isinstance(dep.index, (sympy.Integer, int))
                or dep.stride1_for_last_dim()
                for dep in itertools.chain(
                    node.read_writes.reads, node.read_writes.writes
                )
            ):
                return True
        return False

    @staticmethod
    def reduction_hint(node: Any) -> ReductionHint:
        # [FUNCTION DOC] Static helper: hint for one reduction node. Returns INNER if the node's own
        # hint is not INNER and every read/write dependency is contiguous; otherwise returns
        # the node's own hint. Raises AssertionError if the node is not a reduction.
        if not node.is_reduction():
            raise AssertionError("expected node to be a reduction")
        if node.node.data.reduction_hint != ReductionHint.INNER and all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    def memory_stats(
        self, groups_dict: dict[str, sympy.Expr] | None = None
    ) -> MemoryStats:
        # [FUNCTION DOC] Return (and cache by group sizes) the MemoryStats for this kernel. With no
        # groups_dict, groups are (numel, reduction_numel); with a dict whose keys are exactly
        # "x" and "r0_", those sizes are used; any other keys raise NotImplementedError.
        """Analysis to generate features that can be used in heuristics"""
        if groups_dict is None:
            groups = (self.numel, self.reduction_numel)
        elif groups_dict.keys() == OrderedSet(["x", "r0_"]):
            groups = (groups_dict["x"], groups_dict["r0_"])
        else:
            raise NotImplementedError(f"groups_dict={groups_dict!r}")
        result = self._stats_cache.get(groups)
        if result is None:
            self._stats_cache[groups] = result = MemoryStats.compute(
                MemoryEstimator(self, groups)
            )
        return result


class MemoryEstimator:
    # [CLASS DOC] Simulates codegen over a feature's node schedule to collect the reads and
    # writes (MemoryDeps) per scope: outside_loop, each reduction loop, and the persistent
    # (single-pass) variant. Runs fully in the constructor; results are read as attributes.
    # Tracks store_buffer_names and must_keep_buffers; kernel-local buffers are removed after.
    """
    Estimate various properties of the kernel for use in heuristics.
    We simulate the memory effects of CSE/buffer elimination in codegen.
    """

    kernel_sizes: tuple[sympy.Expr, ...]
    outside_loop: MemoryEstimate
    loops: list[MemoryEstimate]
    persistent: MemoryEstimate
    symbols: list[sympy.Symbol]

    def __init__(self, features: SIMDKernelFeatures, groups: Sequence[sympy.Expr]):
        # [FUNCTION DOC] Initialize estimation state for the given features and group sizes
        # (one index symbol per group), then run simulate_codegen and remove_kernel_local.
        # Side effects: creates index symbols; queries V.graph.
        self.features = features
        self.inside_reduction = features.is_reduction()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.num_reductions_dims = 1
        self.groups = groups
        self.symbols = [make_symbol(SymT.INDEX, i) for i in range(len(groups))]
        # We are doing two estimates simultaneously:
        # 1) the first is a for a non-persistent (aka looped) reduction, using self.outside_loop/self.loops
        # we add an item to loops each corresponding to each reduction loop in the kernel
        # outside_loop is only used for broadcasting or point-wise ops that don't use the reduction dimension
        # 2) the second is for a persistent kernel, using self.persistent
        # persistent kernels don't have loops, so we only have one MemoryEstimate()
        # for point-wise ops the two estimates will be the same, they matter for reductions only
        self.outside_loop = MemoryEstimate()
        self.loops = [MemoryEstimate()]
        self.persistent = MemoryEstimate()
        self.simulate_codegen()
        self.remove_kernel_local()

    def simulate_codegen(self) -> None:
        # [FUNCTION DOC] Walk the node schedule, tracking whether we are inside the reduction loop
        # (DisableReduction/EnableReduction markers toggle it; EnableReduction opens a new loop
        # estimate). For each node, rewrite its loop body into kernel index symbols, then record
        # each MemoryDep read/write into the persistent estimate and the scoped (outside-loop or
        # loop) estimate, skipping reads already satisfied by an earlier write. Raises
        # AssertionError if a schedule entry is not a SchedulerNode. Mutates self.
        from .simd import SIMDKernel

        kernel_size_outside_loop = (*self.groups[:-1], sympy.S.One)
        kernel_size_inside_loop = tuple(self.groups)
        self.kernel_sizes = kernel_size_inside_loop

        for node in self.features.node_schedule:
            if node is DisableReduction:
                self.inside_reduction = False
                self.kernel_sizes = kernel_size_outside_loop
                continue
            elif node is EnableReduction:
                self.inside_reduction = True
                self.kernel_sizes = kernel_size_inside_loop
                self.loops.append(MemoryEstimate())
                continue
            if not isinstance(node, SchedulerNode):
                raise AssertionError(f"expected SchedulerNode, got {type(node)}")
            rw = extract_loop_body_with_args(
                node._body,
                SIMDKernel.map_kernel_groups_to_node_sizes(
                    self.kernel_sizes, node.get_ranges(), self.set_ranges
                ),
                dict(zip(self.symbols, self.kernel_sizes)),
            )

            for dep in rw._reads:
                if not isinstance(dep, MemoryDep):
                    continue
                dep = dep.simplify_with_ranges()
                if not self.persistent.writes.get(dep.name):  # cache miss?
                    self.persistent.reads[dep.name].add(dep)
                # the cache behavior of looped kernels is more complex than the persistent case above
                # some operations are lifted outside the loop (if they don't use the reduction dimension)
                # other operations are inside the loop, and can only be reused within the same loop
                if not (
                    self.outside_loop.writes.get(dep.name)
                    or self.loops[-1].writes.get(dep.name)
                ):
                    self.scope(dep).reads[dep.name].add(dep)
                    if dep.name in self.store_buffer_names and self.loops[-1].reads.get(
                        dep.name
                    ):
                        self.must_keep_buffers.add(dep.name)

            for dep in rw._writes:
                if not isinstance(dep, MemoryDep):
                    continue
                dep = dep.simplify_with_ranges()
                self.store_buffer_names.add(dep.name)
                self.persistent.writes[dep.name].add(dep)
                self.scope(dep).writes[dep.name].add(dep)

    def remove_kernel_local(self) -> None:
        # [FUNCTION DOC] Drop stored buffers that need not hit memory: ones never read persistently
        # and removable through fusion according to the scheduler. They are removed from the
        # persistent estimate and, unless in must_keep_buffers, from outside_loop and all loops.
        # Finally pops the trailing empty loop estimate (pointwise kernels). Mutates self.
        # Remove any kernel-local buffers
        fused_node_names = OrderedSet(
            [n.get_name() for n in self.features.scheduler_nodes()]
        )
        for name in self.store_buffer_names:
            if not self.persistent.reads.get(
                name
            ) and V.graph.scheduler.can_buffer_be_removed_through_fusion(
                name, fused_node_names
            ):
                self.persistent.remove(name)
                if name not in self.must_keep_buffers:
                    # we can also remove this from the looped kernel
                    self.outside_loop.remove(name)
                    for loop in self.loops:
                        loop.remove(name)

        if not self.loops[-1]:
            self.loops.pop()  # for pointwise ops

    def scope(self, dep: MemoryDep) -> MemoryEstimate:
        # [FUNCTION DOC] Return the current loop estimate when inside a reduction and the dep uses a
        # reduction variable or is indirect; otherwise the outside_loop estimate.
        """Determine how a read/write should be categorized"""
        if self.inside_reduction and (
            self.has_reduction_var(dep.index) or dep.is_indirect()
        ):
            return self.loops[-1]
        return self.outside_loop

    def has_reduction_var(self, index: sympy.Expr) -> bool:
        # [FUNCTION DOC] True if the index expression mentions any of the trailing reduction
        # dimension symbols (the last num_reductions_dims symbols).
        for sym in self.symbols[-self.num_reductions_dims :]:
            if isinstance(sym, sympy.Symbol) and sym in index.free_symbols:
                return True
        return False

    def set_ranges(self, *lengths: list[list[sympy.Expr]]) -> list[list[sympy.Expr]]:
        # [FUNCTION DOC] HELPER: Callback passed to the kernel-group mapper. Given one list of node
        # loop lengths per kernel group, returns for each group the index expressions obtained
        # from flattening that group's symbol. Raises AssertionError if the count of
        # length lists differs from the number of kernel sizes.
        if len(self.kernel_sizes) != len(lengths):
            raise AssertionError(
                f"expected len(kernel_sizes) == len(lengths), got "
                f"{len(self.kernel_sizes)} != {len(lengths)}"
            )
        return [
            self.make_flat_range(sym, numel, length)
            for sym, numel, length in zip(self.symbols, self.kernel_sizes, lengths)
        ]

    @staticmethod
    def make_flat_range(
        sym: sympy.Symbol, numel: sympy.Expr, lengths: list[sympy.Expr]
    ) -> list[sympy.Expr]:
        # [FUNCTION DOC] HELPER: Static. Split a flat index symbol into per-dimension index
        # expressions for the given lengths (innermost last). A single length equal to numel
        # returns the symbol itself. Otherwise each dim is a floor division by the running
        # stride when it is statically the outermost remainder, else a modular indexing by
        # its length. Returns expressions in the original length order.
        if len(lengths) == 1 and numel == lengths[0]:
            return [sym]
        divisor = sympy.S.One
        itervars = []
        for length in reversed(lengths):
            if V.graph.sizevars.statically_known_equals(divisor * length, numel):
                expr = FloorDiv(sym, divisor)
            else:
                expr = ModularIndexing(sym, divisor, length)
            itervars.append(expr)
            divisor = divisor * length
        return [*reversed(itervars)]


@dataclasses.dataclass
class MemoryEstimate:
    # [CLASS DOC] Dataclass holding, for one loop scope, mappings from buffer name to the ordered
    # set of MemoryDeps read and written. Truthy when either mapping is non-empty.
    """Tracks the memory usage of a single loop in the generated kernel"""

    reads: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )
    writes: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )

    def remove(self, name: str) -> None:
        # [FUNCTION DOC] Delete the named buffer from both reads and writes; no error if absent.
        self.reads.pop(name, None)
        self.writes.pop(name, None)

    def __bool__(self) -> bool:
        # [FUNCTION DOC] True if there is any recorded read or write.
        return bool(self.reads or self.writes)

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Debug string listing all read and write deps flattened across buffers.
        return f"""MemoryEstimate(
            reads={[*itertools.chain.from_iterable(self.reads.values())]!r},
            writes={[*itertools.chain.from_iterable(self.writes.values())]!r}
        )"""


@dataclasses.dataclass
class StatsForDim:
    # [CLASS DOC] Dataclass of load/store statistics for one kernel block dimension: per-thread
    # op counts and bytes split into contiguous, broadcast and non-contiguous, plus total
    # kernel bytes (symbolic) split into contiguous-or-broadcast and non-contiguous. Supports
    # addition and derived totals.
    """Memory usage stats for a block dimension in the generated kernel (different from user dimensions)"""

    # the number of load/store ops
    count_per_thread_contiguous: int = 0
    count_per_thread_broadcast: int = 0
    count_per_thread_non_contiguous: int = 0  # excludes broadcast

    # total bytes in each load/store op for a single element
    bytes_per_thread_contiguous: int = 0
    bytes_per_thread_broadcast: int = 0
    bytes_per_thread_non_contiguous: int = 0  # excludes broadcast

    # total bytes read by entire kernel
    bytes_contiguous_or_broadcast: sympy.Expr = sympy.S.Zero
    bytes_non_contiguous: sympy.Expr = sympy.S.Zero

    def __add__(self, other: typing.Self) -> StatsForDim:
        # [FUNCTION DOC] Return a new StatsForDim with every field summed with the other's.
        return StatsForDim(
            count_per_thread_contiguous=self.count_per_thread_contiguous
            + other.count_per_thread_contiguous,
            count_per_thread_broadcast=self.count_per_thread_broadcast
            + other.count_per_thread_broadcast,
            count_per_thread_non_contiguous=self.count_per_thread_non_contiguous
            + other.count_per_thread_non_contiguous,
            bytes_per_thread_contiguous=self.bytes_per_thread_contiguous
            + other.bytes_per_thread_contiguous,
            bytes_per_thread_broadcast=self.bytes_per_thread_broadcast
            + other.bytes_per_thread_broadcast,
            bytes_per_thread_non_contiguous=self.bytes_per_thread_non_contiguous
            + other.bytes_per_thread_non_contiguous,
            bytes_contiguous_or_broadcast=self.bytes_contiguous_or_broadcast
            + other.bytes_contiguous_or_broadcast,
            bytes_non_contiguous=self.bytes_non_contiguous + other.bytes_non_contiguous,
        )

    @property
    def count_per_thread(self) -> int:
        # [FUNCTION DOC] Property: total load/store ops per thread (contiguous + broadcast + non-contiguous).
        return (
            self.count_per_thread_contiguous
            + self.count_per_thread_broadcast
            + self.count_per_thread_non_contiguous
        )

    @property
    def bytes_per_thread(self) -> int:
        # [FUNCTION DOC] Property: total bytes per thread (contiguous + broadcast + non-contiguous).
        return (
            self.bytes_per_thread_contiguous
            + self.bytes_per_thread_broadcast
            + self.bytes_per_thread_non_contiguous
        )

    @property
    def bytes(self) -> sympy.Expr:
        # [FUNCTION DOC] Property: total kernel bytes, contiguous-or-broadcast plus non-contiguous.
        return self.bytes_contiguous_or_broadcast + self.bytes_non_contiguous

    @property
    def contiguous_score(self) -> float:
        # [FUNCTION DOC] Property: 1.0 minus the fraction of per-thread ops that are non-contiguous
        # (denominator floored at 1). 1.0 means fully contiguous/broadcast.
        return 1.0 - self.count_per_thread_non_contiguous / max(
            self.count_per_thread, 1
        )


@dataclasses.dataclass
class StatsForLoop:
    # [CLASS DOC] Dataclass with per-thread load/store op count and byte count for one loop.
    """Memory usage stats for single loop in the generated kernel"""

    # load/store ops
    count_per_thread: int = 0
    bytes_per_thread: int = 0

    def __add__(self, other: typing.Self) -> StatsForLoop:
        # [FUNCTION DOC] Return a new StatsForLoop with counts and bytes summed.
        return StatsForLoop(
            count_per_thread=self.count_per_thread + other.count_per_thread,
            bytes_per_thread=self.bytes_per_thread + other.bytes_per_thread,
        )


@dataclasses.dataclass
class StatsForReadsOrWrites:
    # [CLASS DOC] Dataclass of memory stats for reads, writes, or both: a StatsForDim per kernel
    # dimension, a StatsForLoop per loop, and overall symbolic byte totals split into
    # contiguous-or-broadcast and non-contiguous. Built by the compute classmethod.
    """Memory usage stats that are collected for reads/writes/both"""

    dim: list[StatsForDim]
    loop: list[StatsForLoop]
    # total bytes contiguous in any dimension
    bytes_contiguous_or_broadcast: sympy.Expr = sympy.S.Zero
    bytes_non_contiguous: sympy.Expr = sympy.S.Zero

    def __add__(self, other: typing.Self) -> StatsForReadsOrWrites:
        # [FUNCTION DOC] Combine two stats element-wise (per dim, per loop, byte totals). Raises
        # AssertionError if the dim or loop list lengths differ.
        # KNOWN ISSUE: bytes_contiguous_or_broadcast adds self to itself instead of to the
        # other's value, so the other operand's contiguous bytes are ignored and self's doubled.
        if len(self.dim) != len(other.dim):
            raise AssertionError(
                f"expected len(self.dim) == len(other.dim), got "
                f"{len(self.dim)} != {len(other.dim)}"
            )
        if len(self.loop) != len(other.loop):
            raise AssertionError(
                f"expected len(self.loop) == len(other.loop), got "
                f"{len(self.loop)} != {len(other.loop)}"
            )
        return StatsForReadsOrWrites(
            dim=[a + b for a, b in zip(self.dim, other.dim)],
            loop=[a + b for a, b in zip(self.loop, other.loop)],
            bytes_contiguous_or_broadcast=self.bytes_contiguous_or_broadcast
            + self.bytes_contiguous_or_broadcast,
            bytes_non_contiguous=self.bytes_non_contiguous + other.bytes_non_contiguous,
        )

    @property
    def count_per_thread(self) -> int:
        # [FUNCTION DOC] Property: per-thread op count taken from the first dimension's stats.
        return self.dim[0].count_per_thread

    @property
    def bytes_per_thread(self) -> int:
        # [FUNCTION DOC] Property: per-thread bytes taken from the first dimension's stats.
        return self.dim[0].bytes_per_thread

    @property
    def bytes(self) -> sympy.Expr:
        # [FUNCTION DOC] Property: total bytes, contiguous-or-broadcast plus non-contiguous.
        return self.bytes_contiguous_or_broadcast + self.bytes_non_contiguous

    @classmethod
    def compute(
        cls,
        loop_deps: list[dict[str, OrderedSet[MemoryDep]]],
        index_symbols: list[sympy.Symbol],
    ) -> typing.Self:
        # [FUNCTION DOC] Build stats from per-loop mappings of buffer name to MemoryDeps. For each
        # buffer and dep, classify each index symbol dimension by the index stride (1 means
        # contiguous, 0 and not indirect means broadcast, else non-contiguous), accumulate
        # per-thread counts/bytes by dtype itemsize, and estimate total bytes (sum of dep
        # numels, capped by the buffer numel when several deps). When there are several loops,
        # the first (outside-loop) entry is added into each later loop's stats. Raises
        # AssertionError on an empty dep set. Queries V.graph for dtypes and sizes.
        ndim = len(index_symbols)
        result = cls(dim := [StatsForDim() for _ in range(ndim)], [])
        for dep_group in loop_deps:
            result.loop.append(loop_stats := StatsForLoop())
            for name, deps in dep_group.items():
                if not deps:
                    raise AssertionError(f"expected non-empty deps for {name}")
                contiguous_or_broadcast = [True] * ndim
                numel = sympy.S.Zero
                itemsize = V.graph.get_dtype(name).itemsize
                loop_stats.count_per_thread += len(deps)
                loop_stats.bytes_per_thread += itemsize * len(deps)
                for dep in deps:
                    strides: list[sympy.Expr] = V.graph.sizevars.stride_vars(
                        dep.index, index_symbols
                    )
                    for i in range(ndim):
                        if V.graph.sizevars.statically_known_equals(strides[i], 1):
                            dim[i].count_per_thread_contiguous += 1
                            dim[i].bytes_per_thread_contiguous += itemsize
                        elif (
                            V.graph.sizevars.statically_known_equals(strides[i], 0)
                            and not dep.is_indirect()
                        ):
                            dim[i].count_per_thread_broadcast += 1
                            dim[i].bytes_per_thread_broadcast += itemsize
                        else:
                            dim[i].count_per_thread_non_contiguous += 1
                            dim[i].bytes_per_thread_non_contiguous += itemsize
                            contiguous_or_broadcast[i] = False
                    numel += dep.get_numel()
                if len(deps) > 1:
                    # can't read more elements than exist in the buffer
                    numel = Min(numel, V.graph.get_numel(name))
                nbytes = numel * itemsize
                for i in range(ndim):
                    if contiguous_or_broadcast[i]:
                        dim[i].bytes_contiguous_or_broadcast += nbytes
                    else:
                        dim[i].bytes_non_contiguous += nbytes
                if any(contiguous_or_broadcast):
                    result.bytes_contiguous_or_broadcast += nbytes
                else:
                    result.bytes_non_contiguous += nbytes
        if len(result.loop) > 1:
            # the first loop represent the "outside of the loop" compute which could be long lived
            result.loop = [result.loop[0] + x for x in result.loop[1:]]
        return result


@dataclasses.dataclass
class StatsForKernelType:
    # [CLASS DOC] Dataclass bundling StatsForReadsOrWrites for reads, writes, and their combined
    # memory total, for one kernel style (persistent or looped).
    """Memory usage stats that are collected for both persistent and looped kernels"""

    reads: StatsForReadsOrWrites
    writes: StatsForReadsOrWrites
    memory: StatsForReadsOrWrites

    @classmethod
    def compute(
        cls, loops: list[MemoryEstimate], estimator: MemoryEstimator
    ) -> typing.Self:
        # [FUNCTION DOC] Compute read stats and write stats from the given loop estimates using the
        # estimator's index symbols, and set memory to their sum.
        reads = StatsForReadsOrWrites.compute(
            [loop.reads for loop in loops], estimator.symbols
        )
        writes = StatsForReadsOrWrites.compute(
            [loop.writes for loop in loops], estimator.symbols
        )
        return cls(
            reads=reads,
            writes=writes,
            memory=reads + writes,
        )


@dataclasses.dataclass
class MemoryStats:
    # [CLASS DOC] Dataclass holding StatsForKernelType for the persistent and the looped
    # variant of a kernel; computed from a MemoryEstimator.
    """Memory usage stats collected for each generated kernel"""

    persistent: StatsForKernelType
    looped: StatsForKernelType

    def get(self, persistent: bool) -> StatsForKernelType:
        # [FUNCTION DOC] Return the persistent stats if persistent is true, else the looped stats.
        return self.persistent if persistent else self.looped

    @classmethod
    def compute(cls, estimator: MemoryEstimator) -> typing.Self:
        # [FUNCTION DOC] Build MemoryStats from an estimator: persistent stats from its single
        # persistent estimate; looped stats from outside_loop plus loops, reusing the persistent
        # object when there is one loop and not both outside-loop and loop contents.
        persistent = StatsForKernelType.compute([estimator.persistent], estimator)
        if len(estimator.loops) == 1 and not (
            estimator.outside_loop and estimator.loops[0]
        ):
            looped = persistent  # loops/persistent is the same in this common case
        else:
            looped = StatsForKernelType.compute(
                [estimator.outside_loop, *estimator.loops], estimator
            )
        return cls(
            persistent=persistent,
            looped=looped,
        )
