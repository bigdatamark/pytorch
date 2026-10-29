# [MODULE DOC] Inductor scheduler: turns the lowered IR buffers/operations into a dependency graph of
# scheduler nodes (SchedulerNode, ExternKernelSchedulerNode, NopKernelSchedulerNode, FusedSchedulerNode,
# GroupedSchedulerNode, ForeachKernelSchedulerNode, ...), decides which nodes to fuse, reorders them
# (memory/comm aware passes), plans buffer lifetimes/reuse, and drives backend codegen via the
# Scheduler class. Also holds fusion-decision helpers (MixOrderReduction, NestedReduction, FusionResult),
# memory/runtime estimation, graph-partition (cudagraph) support and kernel-metadata utilities.
# Import-time side effects: creates artifact loggers (fusion, loop_ordering, compute_dependencies,
# cudagraphs) and module-level type aliases/constants only.
from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import itertools
import logging
import math
import operator
import os
import pprint
import sys
import textwrap
import time
import traceback
import typing
from collections import Counter, defaultdict
from concurrent.futures import as_completed, Future
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    overload,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
)
from typing_extensions import ParamSpec

from torch.utils._ordered_set import OrderedSet

from .ir import ComputedBuffer, Pointwise


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from types import ModuleType

    from torch._inductor.codegen.wrapper import EnterCudaStreamContextLine

    from .codegen.simd import MemoryCoalescing
    from .codegen.wrapper import PythonWrapperCodegen
    from .tiling_utils import CoalesceVarAnalysis

import sympy

import torch
import torch._inductor.async_compile
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.autotune_process import use_pipelined_autotuning
from torch._inductor.codecache import LambdaFuture, PyCodeCache
from torch._inductor.ir import TritonTemplateCallerBase
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch._inductor.stream_utils import get_stream_name
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._triton import has_triton

from . import comms, config, config_comms, dependencies, ir, metrics
from .analyze_preserves_zero_mask import can_codegen_without_upcasts
from .codegen.common import BackendFeature, get_scheduling_for_device, Kernel
from .comm_analysis import (
    estimate_nccl_collective_runtime,
    estimate_nccl_collective_runtime_nccl_estimator,
)
from .dependencies import Dep, MemoryDep, StarDep, WeakDep
from .exc import GPUTooOldForTriton, TritonMissing
from .fx_utils import count_flops_fx
from .ir import (
    assign_origin_node,
    get_device_type,
    GraphPartitionSignature,
    MultiOutput,
    MultiOutputLayout,
    NoneLayout,
)
from .loop_body import LoopBody
from .memory import MemoryPlanningInfoForBuffer, MemoryPlanningInfoForNode
from .runtime.hints import DeviceProperties, ReductionHint
from .runtime.runtime_utils import green_text, is_power_of_2, red_text
from .sizevars import SimplifyIndexing
from .utils import (
    _unstable_customized_partition_wrapper,
    cache_on_self,
    cache_on_self_and_args,
    cmp,
    decompose_index,
    device_need_guard,
    get_current_backend,
    get_device_tflops,
    get_dtype_size,
    get_gpu_dram_gbps,
    get_op_names,
    GraphPartitionMap,
    IndentedBuffer,
    is_collective,
    is_cudagraph_unsafe_op,
    is_gpu,
    is_multi_outputs_template,
    is_output_of_multi_outputs_template,
    is_wait,
    sympy_product,
    sympy_subs,
)
from .virtualized import V


log = logging.getLogger(__name__)


def _real_dep_names(deps: OrderedSet[Dep]) -> OrderedSet[str]:
    # [FUNCTION DOC] HELPER: Returns the set of buffer names in deps that are real reads/writes, skipping WeakDep entries
    # (which only encode ordering). Result preserves iteration order.
    """Names of real reads/writes, excluding WeakDep (ordering-only deps that
    do not actually read or write the buffer)."""
    return OrderedSet(dep.name for dep in deps if not isinstance(dep, WeakDep))


fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")
loop_ordering_log = torch._logging.getArtifactLogger(__name__, "loop_ordering")
compute_dependencies_log = torch._logging.getArtifactLogger(
    __name__, "compute_dependencies"
)
cudagraphs_log = torch._logging.getArtifactLogger(__name__, "cudagraphs")

PartitionType: TypeAlias = list["BaseSchedulerNode"]
_T = TypeVar("_T")
_P = ParamSpec("_P")
_FLATTENED_READ_VAR = sympy.Dummy("flattened_read", integer=True, nonnegative=True)
_REINDEXING_FUSION_LAUNCH_OVERHEAD_NS = 1_000


@dataclasses.dataclass
class FusionResult:
    # [CLASS DOC] Result of a fusion legality query: either an immediate boolean decision (should_fuse) or a
    # deferred callable_fn (with optional future, e.g. async benchmarking) that yields the decision later.
    # Invariant: exactly one of should_fuse / callable_fn is set; enforced in __post_init__.
    # Factory classmethods: fuse (immediate) and from_callable (deferred).
    should_fuse: bool | None = None
    callable_fn: Callable[[], bool] | None = None
    future: LambdaFuture | None = None

    def __post_init__(self):
        # [FUNCTION DOC] Validates the invariant that exactly one of should_fuse and callable_fn is provided; raises
        # AssertionError otherwise.
        if not ((self.should_fuse is not None) ^ (self.callable_fn is not None)):
            raise AssertionError(
                "Fusion result should contain either fusion decision or callable_fn, not both"
            )

    @classmethod
    def fuse(cls, should_fuse: bool):
        # [FUNCTION DOC] Builds an immediate FusionResult carrying the boolean decision should_fuse.
        return FusionResult(should_fuse=should_fuse)

    @classmethod
    def from_callable(
        cls, callable_fn: Callable[[], bool], future: LambdaFuture | None = None
    ):
        # [FUNCTION DOC] Builds a deferred FusionResult holding callable_fn (returns bool when evaluated) and an optional
        # future that may already be computing the answer.
        return FusionResult(callable_fn=callable_fn, future=future)


@dataclasses.dataclass
class PendingFusion:
    # [CLASS DOC] Record of a fusion whose decision is deferred: the two candidate nodes, the callable that
    # decides fusion, and an optional future for an in-flight computation of that decision.
    callable_fn: Callable[[], bool]
    node1: BaseSchedulerNode
    node2: BaseSchedulerNode
    future: LambdaFuture | None = None

    def get_fusion_nodes(self) -> tuple[BaseSchedulerNode, BaseSchedulerNode]:
        # [FUNCTION DOC] Returns the pair (node1, node2) of nodes this pending fusion concerns.
        return (self.node1, self.node2)


class _LocalEntry(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple row for the memory-aware combo gate's slice: current step (cur), original
    # index used as tie breaker (baseline), and the scheduler node.
    """One row of the post-rewrite slice the gate builds.

    `cur` is the step the node currently wants to run at, `baseline`
    is its original index (used to break ties when sorting), and
    `node` is the slice member itself.
    """

    cur: int
    baseline: int
    node: BaseSchedulerNode


@dataclasses.dataclass(slots=True)
class ComboKernelMemoryContext:
    # [CLASS DOC] Mutable state shared by the memory-aware combo-kernel gate: graph output names, node to original
    # index mapping, baseline peak memory, running peak after accepted combos (threshold still compares against
    # baseline), and per-step live bytes of the original schedule.
    """Shared state used by the memory-aware combo gate.

    Candidate windows are evaluated independently against the original
    schedule. Earlier accepted combos only contribute through `running_peak`,
    which caps the cumulative peak drift from the original graph.
    """

    graph_outputs: OrderedSet[str]
    node_to_idx: dict[BaseSchedulerNode, int]
    baseline_peak: int = 0
    # Running peak after earlier accepts. Bumped on each accept; the
    # gate's threshold check still compares against `baseline_peak`
    # (the original) so total drift is capped.
    running_peak: int = 0
    # Live bytes before each step in the original schedule. Combo windows
    # are evaluated independently, so a window's entry memory is the
    # precomputed baseline live-in at `region_start`.
    baseline_live_before: list[int] = dataclasses.field(default_factory=list)


def _is_gpu_triton_backend(
    node1: BaseSchedulerNode,
    node2: BaseSchedulerNode,
) -> bool:
    # [FUNCTION DOC] HELPER: True when both nodes are GPU nodes on a cuda or xpu device whose current backend is
    # triton; False otherwise. Uses the device of node1.
    if not node1.is_gpu() or not node2.is_gpu():
        return False
    device_type = node1.get_device().type  # type: ignore[union-attr]
    return (
        device_type in ("cuda", "xpu") and get_current_backend(device_type) == "triton"
    )


class MixOrderReduction:
    # [CLASS DOC] Namespace of static/class methods deciding whether two reductions that reduce over different
    # dimensions of the same input (mixed loop orders, e.g. row-wise vs column-wise) can be fused into a
    # single "mix order reduction" kernel. Holds no instance state.
    """
    This class contains utility functions to decide if we should fuse reductions
    reducing across different dimensions of the same input tensor.
    """

    @staticmethod
    def is_split_reduction(node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] True if node is a reduction and every reduction ComputedBuffer sub-node was produced by
        # split-reduction (has a _split_size). Vacuously true for reductions with no such sub-nodes.
        return node.is_reduction() and all(
            subnode.node._split_size is not None
            for subnode in node.get_nodes()
            if isinstance(subnode, SchedulerNode)
            and subnode.is_reduction()
            and isinstance(subnode.node, ComputedBuffer)
        )

    @classmethod
    def get_numel_rnumel(cls, node: BaseSchedulerNode) -> tuple[sympy.Expr, sympy.Expr]:
        # [FUNCTION DOC] Returns (numel, rnumel) for node. For split reductions, derives them from the original (pre-split)
        # ranges of each reduction sub-node, simplified, and checks all sub-nodes agree (raises AssertionError on
        # mismatch or missing data); otherwise returns node.group[1] directly.
        if cls.is_split_reduction(node):
            xnumel = None
            rnumel = None
            for subnode in node.get_nodes():
                if not (
                    isinstance(subnode, SchedulerNode)
                    and subnode.is_reduction()
                    and isinstance(subnode.node, ComputedBuffer)
                ):
                    continue

                if subnode.node._original_ranges is None:
                    raise AssertionError(
                        "expected subnode.node._original_ranges to be set"
                    )
                curxnumel = V.graph.sizevars.simplify(
                    sympy_product(subnode.node._original_ranges)
                )
                if subnode.node._original_reduction_ranges is None:
                    raise AssertionError(
                        "expected subnode.node._original_reduction_ranges to be set"
                    )
                currnumel = V.graph.sizevars.simplify(
                    sympy_product(subnode.node._original_reduction_ranges)
                )

                if xnumel is None:
                    xnumel = curxnumel
                    rnumel = currnumel
                else:
                    if not V.graph.sizevars.statically_known_equals(xnumel, curxnumel):
                        raise AssertionError(f"{xnumel} v.s. {curxnumel}")
                    if not V.graph.sizevars.statically_known_equals(rnumel, currnumel):
                        raise AssertionError(f"{rnumel} v.s. {currnumel}")

            if xnumel is None:
                raise AssertionError("expected xnumel to be set")
            return (xnumel, rnumel)
        else:
            return node.group[1]  # type: ignore[return-value]

    @classmethod
    def has_mix_reduction_orders(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] True when both nodes have (numel, rnumel) groups of length two, differing, and one is the reverse
        # of the other, i.e. they reduce along swapped dimensions.
        g1 = cls.get_numel_rnumel(node1)
        g2 = cls.get_numel_rnumel(node2)

        if len(g1) != 2 or len(g2) != 2 or g1 == g2:
            return False

        return tuple(g1) == tuple(reversed(g2))

    @classmethod
    def _is_full_access(cls, buf: str, node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] HELPER: True if node reads buf with a full (non-broadcast) access: the read index uses all loop
        # variables, or the access size equals the loop iteration size. False if buf is not read as a MemoryDep.
        # Raises AssertionError if var_ranges are missing and node is not a fused node.
        """
        The access to 'buf' is not a broadcast access.
        """
        found_dep = None
        for dep in node.read_writes.reads:
            if isinstance(dep, MemoryDep) and dep.name == buf:
                found_dep = dep
                break

        if not found_dep:
            return False

        index = found_dep.index
        var_ranges = node.read_writes.var_ranges

        if not var_ranges:
            if not isinstance(node, FusedSchedulerNode):
                raise AssertionError(f"{type(node)}")
            var_ranges = node.snodes[0].read_writes.var_ranges

        if not var_ranges:
            raise AssertionError("expected var_ranges to be non-empty")
        if not (OrderedSet(var_ranges) - OrderedSet(index.free_symbols)):
            return True

        # cases that happen after merging loops:
        #   MemoryDep('arg0_1', c0, {c0: 25165824})])
        #   var_ranges={d0: 32768, d1: 768}
        if V.graph.sizevars.statically_known_equals(
            sympy_product(found_dep.size), sympy_product(var_ranges.values())
        ):
            return True
        return False

    @classmethod
    def get_common_read(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> list[str]:
        # [FUNCTION DOC] Returns the list of buffer names used by both nodes that both access with full (non-broadcast)
        # access.
        out = []
        common_reads = node1.used_buffer_names() & node2.used_buffer_names()
        for buf in common_reads:
            if cls._is_full_access(buf, node1) and cls._is_full_access(buf, node2):
                out.append(buf)

        return out

    @classmethod
    def has_common_read(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] True if the two nodes share at least one fully-accessed buffer.
        return len(cls.get_common_read(node1, node2)) > 0

    @classmethod
    def get_numel(cls, node: BaseSchedulerNode) -> int:
        # [FUNCTION DOC] Returns the total element count (numel * rnumel) of node as an optimization hint integer
        # (fallback 0 for unknown dynamic sizes).
        g1 = cls.get_numel_rnumel(node)
        return V.graph.sizevars.optimization_hint(g1[0] * g1[1], fallback=0)

    @classmethod
    def get_fusion_score(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        # [FUNCTION DOC] Fusion score for a mix-order-reduction pair; currently just the numel of node1 (node2 ignored).
        # node2 is ignored for now
        return cls.get_numel(node1)

    # TODO add a cache
    @classmethod
    def can_fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] Decides whether node1 and node2 can be fused as a mix order reduction. Requires the triton config
        # flag, no cpp_wrapper, GPU triton backend, two reductions without strict sum, not producer/consumer
        # related, swapped reduction orders, a common fully-accessed read, and one contiguous node. In strict mode
        # also needs a big enough workload (numel threshold, nrow at least twice ncol and at least 4096). Further
        # requires persistent-reduction friendly hints, ncol statically at most 16384, the contiguous node not a
        # split reduction, and the other node only sum/prod reductions. May add shape guards via evaluate_expr.
        # Returns bool.
        """
        Check whether we can fuse two reductions with mix loop orders.
        """
        if not config.triton.mix_order_reduction:
            return False

        # TODO: Mix order reduction is not supported with cpp_wrapper yet
        if V.graph.cpp_wrapper:
            return False

        if not _is_gpu_triton_backend(node1, node2):
            return False
        if not node1.is_reduction() or not node2.is_reduction():
            return False
        if node1.has_strict_sum() or node2.has_strict_sum():
            return False

        if (node1.ancestors & node2.get_operation_names()) or (
            node2.ancestors & node1.get_operation_names()
        ):
            # Mix-order reductions are sibling reductions, not producer-consumer.
            return False

        # check for mix reduction orders
        if not cls.has_mix_reduction_orders(node1, node2):
            return False

        # check common buffer accesses
        common_reads = MixOrderReduction.get_common_read(node1, node2)
        if len(common_reads) == 0:
            return False

        if cls.is_contiguous_node(node1):
            contiguous_node, other_node = node1, node2
        elif cls.is_contiguous_node(node2):
            contiguous_node, other_node = node2, node1
        else:
            return False

        g1 = cls.get_numel_rnumel(contiguous_node)
        nrow, ncol = g1

        # in non strict mode, we will skip the non-critical checks
        if not config.triton.mix_order_reduction_non_strict_mode:
            # the fused version has worse perf than non-fused version for
            # small workload. When a workload is small enough, data can be
            # fully cached by L2
            size_thres = 5 * 2**20

            # Call evaluate_expr rather than statically_known_geq since nrow can
            # have dynamic shape in real models.
            # Don't use hint directly since hint can be non-representative.
            if not V.graph.sizevars.evaluate_expr(
                sympy.Ge(nrow * ncol, size_thres),
                size_oblivious=True,
                fallback_value=False,
            ):
                return False

            # We require more more row than columns since
            # 1, we prefer doing persistent reduction for each row
            # 2, we will split the reduction across the rows
            if not V.graph.sizevars.evaluate_expr(
                sympy.Ge(nrow, ncol * 2),
                size_oblivious=True,
                fallback_value=False,
            ):
                return False

            # When nrow is small, ncol should also be small (due to the check
            # above). Thus the entire tensor should be well cached in L2.
            # Mix order reduction is less beneficial.
            if not V.graph.sizevars.evaluate_expr(
                sympy.Ge(nrow, 4096),
                size_oblivious=True,
                fallback_value=False,
            ):
                return False

        # Make sure a persistent reduction will be generated
        if any(
            subnode.node.data.reduction_hint  # type: ignore[union-attr]
            not in (
                ReductionHint.INNER,
                ReductionHint.DEFAULT,
            )
            for subnode in contiguous_node.get_nodes()
            if subnode.is_reduction()
        ):
            return False

        # rnumel so large that we will not generated persistent reduction
        # We don't see real use cases with dynamic ncol. But if we do,
        # we should call evaluete_expr here which adds guards.
        if not V.graph.sizevars.statically_known_leq(ncol, 1024 * 16):
            return False

        if MixOrderReduction.is_split_reduction(contiguous_node):
            return False

        # Other reduction types like max/min is not supported yet.
        # There are no real use case as well.
        out = all(
            subnode.node.get_reduction_type()  # type: ignore[union-attr]
            in {
                "sum",
                "prod",
            }
            for subnode in other_node.get_nodes()
            if subnode.is_reduction()
        )
        return out

    @classmethod
    def are_mix_order_reductions(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Alias of can_fuse: True if the two nodes are fusable mix order reductions.
        return cls.can_fuse(node1, node2)

    @classmethod
    def is_contiguous_node(cls, node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] True if every buffer read by node is loaded contiguously (see is_contiguous_load).
        if not all(
            cls.is_contiguous_load(dep.name, node) for dep in node.read_writes.reads
        ):
            return False
        return True

    @classmethod
    def is_contiguous_load(cls, buf: str, parent_node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] Checks, over all sub-nodes of parent_node (each must be a SchedulerNode, else AssertionError), that
        # loads of buf have last-variable stride 0 or 1 in every index expression. Sub-nodes not loading buf are
        # skipped. Assumes the last loop symbol is the reduction var. Returns False on first violating load.
        from torch._inductor.loop_body import MemoryUsageType

        for node in parent_node.get_nodes():
            if not isinstance(node, SchedulerNode):
                raise AssertionError("expected node to be a SchedulerNode")
            loop_body = node._body
            entries = loop_body.memory_usage[MemoryUsageType.LOAD]
            index_names = [e.index_name for e in entries if e.buffer_name == buf]

            if len(index_names) == 0:
                continue

            # there can be multiple index_names some times
            for index_name in index_names:
                index_expr = loop_body.indexing_exprs[index_name]
                var_ranges = loop_body.var_ranges

                # assumes the final symbol is for reduction
                var_symbols = list(var_ranges.keys())
                stride_vars = V.graph.sizevars.stride_vars(
                    index_expr,
                    var_symbols,
                    var_symbols,
                )

                # stride==0 means a broadcast
                if not (stride_vars[-1] == 0 or stride_vars[-1] == 1):
                    return False
        return True


class NestedReduction:
    # [CLASS DOC] Detects when an outer reduction and a dependent grouped (local) reduction over the same logical
    # elements can be fused in one kernel (e.g. layernorm followed by block amax). Limited to pairs with equal
    # total numel. Constants cap group sizes (inner R and non-inner). Nested enums describe pointwise
    # domains and grouped axes; a frozen dataclass bundles domain context.
    """
    Detects when an outer reduction and a dependent grouped reduction can be
    fused into one kernel. The outer reduction reduces over a large dimension
    (e.g. D) producing per-row statistics; the grouped reduction performs a
    small local reduction over the same logical elements. The grouped reduction
    may re-read the outer reduction's large input, consume its full-resolution
    output, or both.

    This is deliberately limited to same-total-numel pairs:
    both reductions must traverse the same number of logical elements.
    General output-size-reducing nested reductions, including split reductions,
    need different grid ownership and are rejected here.

    Example:
    - layernorm + block amax: amax over groups of G after layer_norm
    """

    MAX_INNER_R_GROUP_SIZE = 512
    MAX_NON_INNER_GROUP_SIZE = 128

    @staticmethod
    def _is_dependent_reduction_pair(
        outer_node: BaseSchedulerNode, grouped_node: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if both nodes are reductions and the grouped node has the outer node's operations among
        # its ancestors (it consumes the outer reduction).
        """Check that the grouped reduction is a consumer of the outer reduction."""
        return (
            outer_node.is_reduction()
            and grouped_node.is_reduction()
            and bool(outer_node.get_operation_names() & grouped_node.ancestors)
        )

    @staticmethod
    def _is_enabled_for(
        outer_node: BaseSchedulerNode, grouped_node: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if nested reduction fusion is enabled by config, not cpp_wrapper, GPU triton backend, and
        # neither node has a strict sum.
        return (
            config.triton.nested_reduction
            and not V.graph.cpp_wrapper
            and _is_gpu_triton_backend(outer_node, grouped_node)
            and not outer_node.has_strict_sum()
            and not grouped_node.has_strict_sum()
        )

    @classmethod
    def is_candidate(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] Cheap filter: True when nodes are enabled, form a dependent reduction pair and have statically
        # different reduction numel (rnumel).
        """Cheap filter for dependent reductions with different reduction sizes."""
        if not cls._is_enabled_for(
            node1, node2
        ) or not cls._is_dependent_reduction_pair(node1, node2):
            return False
        _, (_, rnumel1) = node1.group
        _, (_, rnumel2) = node2.group
        return not V.graph.sizevars.statically_known_equals(rnumel1, rnumel2)

    class PointwiseDomain(enum.Enum):
        # [CLASS DOC] Enum of the domain a pointwise node executes in within the nested pipeline: REDUCED (local
        # reduction output), LOCAL_REDUCTION_INPUT (before local reduce) or PARENT_FULL (outer tile after
        # broadcast-back).
        """
        Where a pointwise node runs in the nested pipeline.

        The local reduction stage has three meaningful domains: its reduced
        output, its input before reducing the local lane, and the outer
        reduction's parent tile after broadcast-back.
        """

        # Local reduction output, e.g. [B, D // G].
        REDUCED = enum.auto()
        # Input domain before reducing the local group, e.g. [B, D // G, G].
        LOCAL_REDUCTION_INPUT = enum.auto()
        # Outer reduction tile after broadcast-back, e.g. [B, D].
        PARENT_FULL = enum.auto()

    class GroupedAxis(enum.Enum):
        # [CLASS DOC] Enum naming which axis of the grouped reduction is reduced: R or X.
        R = enum.auto()
        X = enum.auto()

    @dataclasses.dataclass(frozen=True)
    class PointwiseDomainContext:
        # [CLASS DOC] Frozen record bundling the grouped reduction node, its numel/rnumel, and the local-reduction and
        # parent-full domain shapes used when classifying pointwise nodes.
        grouped_reduction: SchedulerNode
        grouped_numel: sympy.Expr
        grouped_rnumel: sympy.Expr
        local_reduction_domain: tuple[sympy.Expr, ...]
        parent_full_domain: tuple[sympy.Expr, ...]

    @classmethod
    def _get_grouped_reduction_and_size(
        cls, grouped_node: BaseSchedulerNode, grouped_rnumel: sympy.Expr
    ) -> tuple[SchedulerNode, sympy.Integer] | None:
        # [FUNCTION DOC] HELPER: Validates that grouped_node is a single simple reduction (exactly one reduction sub-node that is a
        # SchedulerNode over a ComputedBuffer, with one or two iteration range trees and one reduce range, and a simple
        # reduction type: any, max, min, prod, sum, xor_sum) and that grouped_rnumel simplifies to a positive static
        # integer. Returns (reduction node, group size as an integer) or None if any check fails.
        """Validate the candidate as a single simple grouped reduction."""
        if not grouped_node.is_reduction():
            return None
        reductions = [sn for sn in grouped_node.get_nodes() if sn.is_reduction()]
        if len(reductions) != 1:
            return None
        reduction = reductions[0]
        if not isinstance(reduction, SchedulerNode) or not isinstance(
            reduction.node, ComputedBuffer
        ):
            return None
        iter_ranges, reduce_ranges = reduction.get_ranges()
        # The original tensor may be higher-rank, but scheduler ranges are
        # already collapsed into loop trees. Nested codegen maps the grouped
        # reduction as either [outer_x, r_groups, G] or [x_groups, outer_r, G].
        # A singleton non-reduced tree may be squeezed out; additional trees
        # need explicit axis provenance and are rejected here.
        if len(iter_ranges) not in (1, 2) or len(reduce_ranges) != 1:
            return None
        # Single-value reductions are lowered by reshape + tl.<op>(..., axis).
        # Tuple/stateful reductions such as argmax and Welford need the normal
        # reduction lowering and are rejected before nested codegen.
        if reduction.node.get_reduction_type() not in {
            "any",
            "max",
            "min",
            "prod",
            "sum",
            "xor_sum",
        }:
            return None

        group_size = V.graph.sizevars.simplify(grouped_rnumel)
        if not isinstance(group_size, (int, sympy.Integer)) or int(group_size) < 1:
            return None
        return reduction, sympy.Integer(group_size)

    @classmethod
    def _pointwise_nodes_match_nested_domains(
        cls,
        outer_node: BaseSchedulerNode,
        grouped_node: BaseSchedulerNode,
        domain_context: PointwiseDomainContext,
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if every pointwise node in the outer and grouped nodes can be classified into a nested
        # pipeline domain and each classification has compatible ranges. False if classification fails.
        """Classify pointwise nodes by nested stage and validate their ranges.

        Each pointwise node must run either on the grouped reduction output,
        on the grouped reduction's full local-group domain, or on the outer
        reduction's full parent domain, based on its producer/consumer
        relationship to the grouped reduction.
        """
        pointwise_domains = cls._classify_nested_pointwise_nodes(
            outer_node,
            grouped_node,
            domain_context,
        )
        if pointwise_domains is None:
            return False

        return cls._pointwise_domains_are_compatible(domain_context, pointwise_domains)

    @classmethod
    def _classify_nested_pointwise_nodes(
        cls,
        outer_node: BaseSchedulerNode,
        grouped_node: BaseSchedulerNode,
        domain_context: PointwiseDomainContext,
    ) -> list[tuple[SchedulerNode, PointwiseDomain]] | None:
        # [FUNCTION DOC] HELPER: Classifies pointwise sub-nodes of outer_node that consume an outer reduction result as running in
        # the local-reduction-input domain, then adds the classification of the grouped node's pointwise nodes.
        # Returns the combined list of (SchedulerNode, domain) pairs, or None if any such node is not a
        # SchedulerNode or grouped classification fails.
        outer_pointwise_domains: list[
            tuple[SchedulerNode, NestedReduction.PointwiseDomain]
        ] = []
        # Use ancestor names to find pointwise subnodes inside the outer fused
        # node that consume an outer reduction result.
        outer_reduction_names: OrderedSet[str] = OrderedSet()
        for sn in outer_node.get_nodes():
            if sn.is_reduction():
                outer_reduction_names |= sn.get_operation_names()

        # Full-resolution consumers already fused into the outer reduction feed
        # the local reduction, so they run in that input domain.
        for sn in outer_node.get_nodes():
            if sn.is_reduction():
                continue
            if outer_reduction_names & sn.ancestors:
                if not isinstance(sn, SchedulerNode):
                    return None
                outer_pointwise_domains.append(
                    (sn, cls.PointwiseDomain.LOCAL_REDUCTION_INPUT)
                )

        grouped_pointwise_domains = cls._classify_grouped_pointwise_nodes(
            domain_context,
            grouped_node.get_nodes(),
        )
        if grouped_pointwise_domains is None:
            return None
        return [*outer_pointwise_domains, *grouped_pointwise_domains]

    @classmethod
    def _classify_grouped_pointwise_nodes(
        cls,
        domain_context: PointwiseDomainContext,
        nodes: Sequence[BaseSchedulerNode],
    ) -> list[tuple[SchedulerNode, PointwiseDomain]] | None:
        # [FUNCTION DOC] HELPER: Classifies each non-reduction node as a producer of the grouped reduction (local-reduction-input
        # or reduced domain) or a consumer (parent-full or reduced domain) according to its numel versus the
        # grouped numel and the full numel. Returns a list of (node, domain), or None for non-SchedulerNodes, nodes
        # that are both or neither producer/consumer, or numels matching no domain.
        """Classify pointwise nodes relative to the grouped reduction.

        A node must be on exactly one side of the grouped reduction: either a
        producer feeding its local-group body, or a consumer of its reduced
        output. Its numel then determines whether it runs at reduced,
        grouped-full, or parent-full resolution.
        """
        grouped_reduction = domain_context.grouped_reduction
        reduction_names = grouped_reduction.get_operation_names()
        full_numel = V.graph.sizevars.simplify(
            domain_context.grouped_numel * domain_context.grouped_rnumel
        )
        pointwise_domains: list[
            tuple[SchedulerNode, NestedReduction.PointwiseDomain]
        ] = []
        for sn in nodes:
            if sn.is_reduction():
                continue
            if not isinstance(sn, SchedulerNode):
                return None

            sn_names = sn.get_operation_names()
            is_producer = bool(sn_names & grouped_reduction.ancestors)
            is_consumer = bool(reduction_names & sn.ancestors)
            if is_producer and is_consumer:
                # Supportable by splitting/modeling a multi-stage pointwise,
                # but not as one nested pipeline stage today.
                return None
            if not is_producer and not is_consumer:
                # Supportable as a sidecar, but nested codegen does not yet
                # model an insertion point for unrelated pointwise nodes.
                return None

            full_domain = (
                cls.PointwiseDomain.LOCAL_REDUCTION_INPUT
                if is_producer
                else cls.PointwiseDomain.PARENT_FULL
            )
            _, (sn_numel, _) = sn.group
            if V.graph.sizevars.statically_known_equals(
                sn_numel, domain_context.grouped_numel
            ):
                domain = cls.PointwiseDomain.REDUCED
            elif V.graph.sizevars.statically_known_equals(sn_numel, full_numel):
                domain = full_domain
            else:
                return None
            pointwise_domains.append((sn, domain))
        return pointwise_domains

    @classmethod
    def _pointwise_domains_are_compatible(
        cls,
        domain_context: PointwiseDomainContext,
        pointwise_domains: Sequence[tuple[SchedulerNode, PointwiseDomain]],
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if every (node, domain) pair is range-compatible with the domain context.
        return all(
            cls._pointwise_domain_is_compatible(sn, domain, domain_context)
            for sn, domain in pointwise_domains
        )

    @classmethod
    def _pointwise_domain_is_compatible(
        cls,
        sn: SchedulerNode,
        domain: PointwiseDomain,
        domain_context: PointwiseDomainContext,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Checks one pointwise node: its numel must statically equal the numel expected for its domain
        # (grouped numel for reduced, grouped total for the other two) and its ranges must be compatible (via
        # SIMDKernel.is_compatible) with the domain's expected group shape. Raises AssertionError on an unknown domain.
        from .codegen.simd import SIMDKernel

        iter_ranges, _ = domain_context.grouped_reduction.get_ranges()
        _, (sn_numel, _) = sn.group
        if domain is cls.PointwiseDomain.REDUCED:
            expected_numel = domain_context.grouped_numel
            expected_groups: Sequence[sympy.Expr] = tuple(iter_ranges)
        elif domain is cls.PointwiseDomain.LOCAL_REDUCTION_INPUT:
            expected_numel = V.graph.sizevars.simplify(
                domain_context.grouped_numel * domain_context.grouped_rnumel
            )
            expected_groups = domain_context.local_reduction_domain
        else:
            if domain is not cls.PointwiseDomain.PARENT_FULL:
                raise AssertionError(f"expected PARENT_FULL domain, got {domain}")
            expected_numel = V.graph.sizevars.simplify(
                domain_context.grouped_numel * domain_context.grouped_rnumel
            )
            expected_groups = domain_context.parent_full_domain
        return V.graph.sizevars.statically_known_equals(
            sn_numel, expected_numel
        ) and SIMDKernel.is_compatible(expected_groups, sn.get_ranges())

    @classmethod
    def _min_block_unprofitable_for_kernel(
        cls,
        outer_node: BaseSchedulerNode,
        outer_numel: sympy.Expr,
        outer_rnumel: sympy.Expr,
        *,
        grouped_axis: GroupedAxis,
        group_size: int,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Returns True (unprofitable/unsupported) unless the outer node is a SchedulerNode or fused node whose
        # selected tiling is exactly x and r0_ matching outer numel/rnumel and group_size does not exceed the
        # maximum minimum-block group size for the schedule. Calls SIMDScheduling.select_tiling (may use coalesce
        # analysis per config).
        from .codegen.simd import SIMDScheduling

        if not isinstance(outer_node, (SchedulerNode, FusedSchedulerNode)):
            return True
        coalesce_analysis = (
            outer_node.get_coalesce_analysis()
            if config.triton.coalesce_tiling_analysis
            else None
        )
        node_schedule = list(outer_node.get_nodes())
        tiling = SIMDScheduling.select_tiling(
            node_schedule,
            outer_numel,
            outer_rnumel,
            coalesce_analysis,
        )
        # TODO: fold richer profitability/coalescing policy into this guard.
        # For now this is only a codegen capability check for the min block.
        # The grouped reduction forces a minimum block on the split axis. Today
        # that floor is only modeled for ordinary x/r0 kernels, where x is the
        # outer pointwise axis and r0_ is the inner reduction axis.
        if OrderedSet(tiling) != OrderedSet(("x", "r0_")):
            return True
        if not (
            V.graph.sizevars.statically_known_equals(tiling["x"], outer_numel)
            and V.graph.sizevars.statically_known_equals(tiling["r0_"], outer_rnumel)
        ):
            return True
        return group_size > cls._max_min_block_group_size(node_schedule, grouped_axis)

    @classmethod
    def _max_min_block_group_size(
        cls,
        node_schedule: Sequence[BaseSchedulerNode],
        grouped_axis: GroupedAxis,
    ) -> int:
        # [FUNCTION DOC] HELPER: Returns the max group size supported as a minimum block: MAX_INNER_R_GROUP_SIZE when grouping
        # the R axis and all reductions in the schedule have an INNER reduction hint, else MAX_NON_INNER_GROUP_SIZE.
        from .codegen.simd_kernel_features import SIMDKernelFeatures

        reduction_hints = [
            SIMDKernelFeatures.reduction_hint(sn)
            for sn in node_schedule
            if isinstance(sn, SchedulerNode)
            and sn.is_reduction()
            and isinstance(sn.node, ComputedBuffer)
        ]
        if (
            grouped_axis is cls.GroupedAxis.R
            and reduction_hints
            and all(hint is ReductionHint.INNER for hint in reduction_hints)
        ):
            return cls.MAX_INNER_R_GROUP_SIZE
        return cls.MAX_NON_INNER_GROUP_SIZE

    @classmethod
    def can_fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] Decides whether a dependent outer/grouped reduction pair can be fused as a nested reduction. Returns False
        # if disabled, equal rnumel, grouped node not a single simple reduction, total elements differ, grouped axis
        # cannot be determined or does not divide parent extent, group size is not a power of two, min-block
        # is unprofitable, or pointwise domains mismatch. Raises AssertionError if the pair is not dependent.
        # Node1 is the outer reduction and node2 the grouped one.
        """Check whether a dependent cross-axis reduction pair can be fused."""
        # TODO: enable nested reduction with cpp wrapper after validating the
        # additional autotuning meta (min_xblock / min_rblock).
        if not cls._is_enabled_for(node1, node2):
            return False

        if not cls._is_dependent_reduction_pair(node1, node2):
            raise AssertionError(
                "expected node1 and node2 to be a dependent reduction pair"
            )

        outer_node = node1
        grouped_node = node2
        outer_group: tuple[sympy.Expr, sympy.Expr]
        grouped_group: tuple[sympy.Expr, sympy.Expr]
        _, outer_group = outer_node.group  # pyrefly: ignore [bad-assignment]
        _, grouped_group = grouped_node.group  # pyrefly: ignore [bad-assignment]
        outer_numel, outer_rnumel = outer_group
        grouped_numel, grouped_rnumel = grouped_group
        if V.graph.sizevars.statically_known_equals(outer_rnumel, grouped_rnumel):
            return False

        # The grouped candidate must be one small block-local reduction.
        # We specialize the grouped reduction on that block size, so it must
        # simplify to an exact static int. Multiple reductions in the grouped
        # candidate (e.g. amax AND sum from the same input) are not supported.
        grouped_reduction_info = cls._get_grouped_reduction_and_size(
            grouped_node, grouped_rnumel
        )
        if grouped_reduction_info is None:
            return False

        # Total-element equality also implies divisibility of the grouped
        # parent extent by group_size (grouped_total factors through
        # FloorDiv(extent, group_size) * group_size). No separate Mod check
        # is needed. The sizevars divisible set records Mod(s, s//G) from
        # the view op, not Mod(s, G), so statically_known_equals on
        # Mod(extent, group_size) would fail with dynamic shapes.
        # TODO: teach sizevars to infer Mod(s, G)==0 from Mod(s, s//G)==0
        outer_total = V.graph.sizevars.simplify(outer_numel * outer_rnumel)
        grouped_total = V.graph.sizevars.simplify(grouped_numel * grouped_rnumel)
        if not V.graph.sizevars.statically_known_equals(outer_total, grouped_total):
            return False

        grouped_reduction, group_size = grouped_reduction_info

        grouped_axis = cls.get_grouped_axis(
            grouped_reduction,
            outer_numel,
            outer_rnumel,
            group_size,
            outer_node=outer_node,
        )
        if grouped_axis is None:
            return False
        parent_grouped_axis = (
            outer_rnumel if grouped_axis is cls.GroupedAxis.R else outer_numel
        )
        iter_ranges, _ = grouped_reduction.get_ranges()
        if len(iter_ranges) == 2:
            grouped_axis_groups = (
                iter_ranges[1] if grouped_axis is cls.GroupedAxis.R else iter_ranges[0]
            )
            if not V.graph.sizevars.statically_known_equals(
                FloorDiv(parent_grouped_axis, group_size), grouped_axis_groups
            ):
                return False
        elif not V.graph.sizevars.statically_known_equals(
            sympy.Mod(parent_grouped_axis, group_size), 0
        ):
            return False
        group_size_int = int(group_size)
        if not (1 <= group_size_int and is_power_of_2(group_size_int)):
            return False
        if cls._min_block_unprofitable_for_kernel(
            outer_node,
            outer_numel,
            outer_rnumel,
            grouped_axis=grouped_axis,
            group_size=group_size_int,
        ):
            return False
        iter_ranges, reduce_ranges = grouped_reduction.get_ranges()
        domain_context = cls.PointwiseDomainContext(
            grouped_reduction=grouped_reduction,
            grouped_numel=grouped_numel,
            grouped_rnumel=grouped_rnumel,
            local_reduction_domain=(*iter_ranges, *reduce_ranges),
            parent_full_domain=(outer_numel, outer_rnumel),
        )
        if not cls._pointwise_nodes_match_nested_domains(
            outer_node,
            grouped_node,
            domain_context,
        ):
            return False

        return True

    @classmethod
    def get_grouped_axis(
        cls,
        grouped_reduction: SchedulerNode,
        outer_numel: sympy.Expr,
        outer_rnumel: sympy.Expr,
        group_size: sympy.Expr,
        *,
        outer_node: BaseSchedulerNode | None = None,
    ) -> GroupedAxis | None:
        # [FUNCTION DOC] Determines which parent axis (GroupedAxis R or X) the grouped local reduction splits, by comparing the
        # grouped reduction's iteration/reduce ranges with outer numel/rnumel and group_size for the 1D and 2D
        # forms; falls back to loop-body analysis when outer_node is given. Returns the axis or None.
        """Return which parent axis is split by the grouped local reduction."""
        sizevars = V.graph.sizevars
        iter_ranges, reduce_ranges = grouped_reduction.get_ranges()
        if len(iter_ranges) != 2 or len(reduce_ranges) != 1:
            if len(iter_ranges) == 1 and len(reduce_ranges) == 1:
                if not sizevars.statically_known_equals(reduce_ranges[0], group_size):
                    return None
                # [R / G] reduce G: parent X is singleton, so the local group
                # splits the parent R axis.
                if sizevars.statically_known_equals(
                    FloorDiv(outer_rnumel, group_size), iter_ranges[0]
                ) and sizevars.statically_known_equals(outer_numel, 1):
                    return cls.GroupedAxis.R
                # [R] reduce G: parent X is exactly one local group and is
                # reduced away, leaving the parent R axis as the only iterator.
                if sizevars.statically_known_equals(
                    FloorDiv(outer_numel, group_size), 1
                ) and sizevars.statically_known_equals(iter_ranges[0], outer_rnumel):
                    return cls.GroupedAxis.X
            return None
        if not sizevars.statically_known_equals(reduce_ranges[0], group_size):
            return None
        # 2D group-in-R form: [X, R / G] reduce G. Dynamic view guards often
        # preserve the quotient dimension even when Mod(R, G) is not directly
        # provable.
        if sizevars.statically_known_equals(
            iter_ranges[0], outer_numel
        ) and sizevars.statically_known_equals(
            FloorDiv(outer_rnumel, group_size), iter_ranges[1]
        ):
            return cls.GroupedAxis.R
        # 2D group-in-X form: [X / G, R] reduce G.
        if sizevars.statically_known_equals(
            iter_ranges[1], outer_rnumel
        ) and sizevars.statically_known_equals(
            FloorDiv(outer_numel, group_size), iter_ranges[0]
        ):
            return cls.GroupedAxis.X
        if outer_node is not None:
            return cls._get_grouped_axis_from_loop_body(outer_node, grouped_reduction)
        return None

    @classmethod
    def _get_grouped_axis_from_loop_body(
        cls, outer_node: BaseSchedulerNode, grouped_reduction: SchedulerNode
    ) -> GroupedAxis | None:
        # [FUNCTION DOC] HELPER: Disambiguates the grouped axis when sizes are equal by comparing, for buffers loaded by both
        # reductions, the coefficient of the grouped reduce variable against the outer reduce/iter variable
        # coefficients. Returns R or X if all evidence agrees, None if no evidence, ambiguity, conflict or
        # non-single reductions.
        """Use LoopBody iter/reduce vars to disambiguate equal-size axes."""
        from torch._inductor.loop_body import MemoryUsageType

        outer_reductions = [sn for sn in outer_node.get_nodes() if sn.is_reduction()]
        if len(outer_reductions) != 1:
            return None
        outer_reduction = typing.cast(SchedulerNode, outer_reductions[0])
        outer_body = getattr(outer_reduction, "_body", None)
        grouped_body = getattr(grouped_reduction, "_body", None)
        if outer_body is None or grouped_body is None:
            return None

        outer_iter_ranges, outer_reduce_ranges = outer_reduction.get_ranges()
        grouped_iter_ranges, grouped_reduce_ranges = grouped_reduction.get_ranges()
        if len(outer_reduce_ranges) != 1 or len(grouped_reduce_ranges) != 1:
            return None

        def load_exprs_by_name(body: LoopBody) -> dict[str, list[sympy.Expr]]:
            # [FUNCTION DOC] HELPER: Maps each loaded buffer name in a loop body to the list of its indexing expressions.
            result: dict[str, list[sympy.Expr]] = defaultdict(list)
            for entry in body.memory_usage.get(MemoryUsageType.LOAD, ()):
                if entry.buffer_name is not None:
                    result[entry.buffer_name].append(
                        body.indexing_exprs[entry.index_name]
                    )
            return result

        if len(outer_body.reduce_vars) != 1 or len(grouped_body.reduce_vars) != 1:
            return None
        if len(outer_body.iter_vars) != len(outer_iter_ranges) or len(
            grouped_body.iter_vars
        ) != len(grouped_iter_ranges):
            return None

        outer_reads_by_name = load_exprs_by_name(outer_body)
        result: NestedReduction.GroupedAxis | None = None
        grouped_reduce_var = grouped_body.reduce_vars[-1]
        for name, grouped_read_exprs in load_exprs_by_name(grouped_body).items():
            outer_read_exprs = outer_reads_by_name.get(name)
            if not outer_read_exprs:
                continue
            for grouped_read_expr in grouped_read_exprs:
                grouped_coeff = grouped_read_expr.coeff(grouped_reduce_var)
                if grouped_coeff == 0:
                    continue
                for outer_read_expr in outer_read_exprs:
                    outer_reduce_var = outer_body.reduce_vars[-1]
                    outer_reduce_coeff = outer_read_expr.coeff(outer_reduce_var)
                    matches_reduction = grouped_coeff == outer_reduce_coeff
                    matches_iter = any(
                        grouped_coeff == outer_read_expr.coeff(var)
                        for var in outer_body.iter_vars
                    )
                    if matches_reduction == matches_iter:
                        continue
                    candidate = (
                        cls.GroupedAxis.R if matches_reduction else cls.GroupedAxis.X
                    )

                    if result is not None and result != candidate:
                        return None
                    result = candidate
        return result


@dataclasses.dataclass
class SchedulerBuffer:
    # [CLASS DOC] Scheduler-level record of one buffer: owning scheduler, the IR buffer node, the defining
    # operation node (may be None), the list of NodeUser users, and memory planning info. Hashed by buffer name.
    # Provides allocation/free codegen helpers and alias/mutation queries.
    scheduler: Scheduler
    node: ir.Buffer
    defining_op: BaseSchedulerNode | None
    users: list[NodeUser] = dataclasses.field(default_factory=list)
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def defining_op_name(self) -> str:
        # [FUNCTION DOC] Returns the name of the defining operation; raises AssertionError if there is none.
        op = self.defining_op
        if op is None:
            raise AssertionError("expected op to be set")
        return op.get_name()

    def is_ordering_only(self) -> bool:
        # [FUNCTION DOC] Returns whether the underlying buffer only encodes ordering (no real data).
        return self.node.ordering_only

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash based on the underlying buffer name.
        return hash(self.node.name)

    def debug_str(self) -> str:
        # [FUNCTION DOC] Returns a multi-line debug string with buffer type, layout, aliases, mutations and users.
        result = IndentedBuffer()
        name = self.get_name()
        result.writeline(f"{name}: {type(self.node).__name__}")
        result.writeline(f"{name}.layout = {self.node.layout}")
        if self.get_aliases():
            result.writeline(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            result.writeline(f"{name}.mutations = {pformat(self.get_mutations())}")

        if len(self.users) <= 1:
            result.writeline(f"{name}.users = {self.users}")
        else:
            result.writeline(f"{name}.users = [")
            with result.indent(1):
                for user in self.users:
                    result.writeline(f"{user},")
            result.writeline("]")
        return result.getrawvalue()

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the buffer name.
        return self.node.get_name()

    def allocate(self) -> None:
        # [FUNCTION DOC] Emits allocation code through the wrapper codegen if the buffer should be allocated. Aliasing, mutating or
        # comm-buffer outputs get a normal allocation; a buffer written in place of an input of the current kernel
        # (inplace_update_buffers) emits inplace reuse of that input (donated or regular buffer); otherwise a normal
        # allocation. Side effect: writes to V.graph.wrapper_code. Raises AssertionError if node is None.
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        if not self.node.should_allocate():
            return

        if (
            self.node.get_inputs_that_alias_output()
            or self.node.get_mutation_names()
            or isinstance(self.node.get_output_spec(), ir.CommBufferLayout)
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        # hacky check for if V.kernel is a real kernel or NullHandler
        if (
            hasattr(V.kernel, "args")
            and self.get_name() in V.kernel.inplace_update_buffers
        ):
            input_buffer: ir.DonatedBuffer | ir.Buffer
            input_buffer_name = V.kernel.inplace_update_buffers[self.get_name()]
            if input_buffer_name in self.scheduler.name_to_donated_buffer:
                input_buffer = self.scheduler.name_to_donated_buffer[
                    input_buffer_name
                ].node
            else:
                input_buffer = self.scheduler.name_to_buf[input_buffer_name].node
            V.graph.wrapper_code.codegen_inplace_reuse(
                input_buffer,
                self.node,
            )
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self) -> bool:
        # [FUNCTION DOC] True if the buffer can be freed: False for none-layout or multi-output template buffers and for buffers
        # that are graph outputs (used by an OutputNode); raises AssertionError if node is None.
        # There's no real allocated buffer, no need to free it
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        if isinstance(self.node.layout, ir.NoneLayout) or is_multi_outputs_template(
            self.node
        ):
            return False
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def set_users(self, users: list[NodeUser]) -> None:
        # [FUNCTION DOC] Sets the users list after deduplicating by user node identity, merging duplicates via NodeUser merge.
        # deduplicate
        result: dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self) -> Sequence[str]:
        # [FUNCTION DOC] Returns the names of inputs the buffer aliases; raises AssertionError if node is None.
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_inputs_that_alias_output()

    def get_mutations(self) -> Sequence[str]:
        # [FUNCTION DOC] Returns the names of buffers this buffer mutates; raises AssertionError if node is None.
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_mutation_names()

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Returns the device of the buffer's output spec.
        return self.node.get_output_spec().get_device()


@dataclasses.dataclass
class SchedulerDonatedBuffer(SchedulerBuffer):
    # [CLASS DOC] SchedulerBuffer for a donated (graph input) buffer; defining_op defaults to None.
    defining_op: BaseSchedulerNode | None = None


class BaseSchedulerNode:
    # [CLASS DOC] Base class of all scheduler nodes. Tracks the owning scheduler, the wrapped IR operation, read/write
    # dependencies and unmet dependencies, ancestors (operation names), output SchedulerBuffers (also by name),
    # input distances, min/max order in the scheduler node list (for grouped nodes), last_usage buffers freed
    # after the node, mutation renames, and an optional runtime override. Subclasses supply the concrete behavior.
    ancestors: OrderedSet[str]
    group: tuple[torch.device, tuple[tuple[sympy.Expr, ...], ...]]
    last_usage: OrderedSet[str]
    # .min_order and .max_order are only relevant for "grouped" nodes such as FusedSchedulerNode.
    # e.g. if the FusedSchedulerNode includes nodes (op_1, op_2, op_3), and op_X is X-th node
    # in `self.scheduler.nodes`, then for this FusedSchedulerNode, .min_order is 1 and .max_order is 3.
    # For non-"grouped" nodes (i.e. regular SchedulerNode),
    # .min_order = .max_order = X if this node is X-th node in `self.scheduler.nodes`.
    min_input_distance: int
    max_input_distance: int
    min_order: int
    max_order: int
    mpi_node: MemoryPlanningInfoForNode
    mutation_renames: dict[str, str]
    node: ir.Operation | None = None
    outputs: list[SchedulerBuffer]
    outputs_by_name: dict[str, SchedulerBuffer]
    override_estimated_runtime: float | None = None
    read_writes: dependencies.ReadWrites
    unmet_dependencies: OrderedSet[Dep]
    written: bool = False

    def __init__(self, scheduler: Scheduler) -> None:
        # [FUNCTION DOC] Stores the scheduler and a default debug-device-string callback returning an empty list.
        self.scheduler: Scheduler = scheduler
        self.debug_device_str: Callable[[BaseSchedulerNode], list[str]] = (
            lambda *args, **kwargs: []
        )

    def _init_from_node(self, node: ir.Operation) -> None:
        # [FUNCTION DOC] HELPER: Initializes per-node state from an IR operation: stores the node, empty ancestors, zero input
        # distances, empty last_usage, written False, creates one SchedulerBuffer per operation output (list and
        # by-name map) and an empty mutation_renames map (limited to deps relevant to this node).
        self.node = node
        self.ancestors = OrderedSet()
        self.min_input_distance = 0
        self.max_input_distance = 0
        self.last_usage = OrderedSet[
            str
        ]()  # buffers that won't be used after this kernel
        self.written = False
        self.outputs = [
            SchedulerBuffer(
                scheduler=self.scheduler,
                node=output,
                defining_op=self,
            )
            for output in node.get_outputs()
        ]
        self.outputs_by_name = {buf.get_name(): buf for buf in self.outputs}

        # mutation_renames for the current node. Due to potential
        # more mutations happening later, this can be different
        # to Scheduler.mutation_renames. Also this dict should be small
        # since only mutation information relevant to the deps for this
        # node is stored here.
        self.mutation_renames = {}

    def __repr__(self) -> str:
        # [FUNCTION DOC] Short repr showing the class name and the node name.
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self) -> str:
        # [FUNCTION DOC] Builds a longer multi-line debug string for trace logs: writes, unmet and met dependencies, input
        # distances and each output's debug string, plus debug_str_extra (errors there are logged as warnings and
        # ignored). Returns the string without trailing whitespace.
        """Longer form printout for trace logs"""
        name = self.get_name()
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({type(getattr(self, "node", None)).__name__})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.min_input_distance = {self.min_input_distance}
{name}.max_input_distance = {self.max_input_distance}
{name}.outputs = [
        """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()

    def debug_str_extra(self) -> str:
        # [FUNCTION DOC] Hook for subclasses to append extra debug text; base returns an empty string.
        return ""

    def _debug_str_for_device(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Calls the debug_device_str callback on this node, returning a list of device-specific debug lines.
        return self.debug_device_str(self)

    def debug_str_short(self) -> str:
        # [FUNCTION DOC] Returns a short debug string: the node repr plus, for pointwise or reduction IR data, the formatted
        # data description.
        maybe_data = getattr(self.node, "data", None)
        data_str = ""
        if isinstance(maybe_data, torch._inductor.ir.Pointwise):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_size()], shorten=False, multiline=False
            )
        elif isinstance(maybe_data, torch._inductor.ir.Reduction):
            data_str = ", " + maybe_data.str_helper(
                [maybe_data.get_reduction_size(), maybe_data.get_reduction_type()],
                shorten=False,
                multiline=False,
            )
        return f"{self}{data_str}"

    def log_details(self) -> None:
        # [FUNCTION DOC] Logs (info level) the node, its unmet dependencies and its writes.
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> bool:
        # [FUNCTION DOC] Hook to reorder loops of this node to match a dependency pair of another node; base implementation does
        # nothing and returns False (no reordering done).
        return False

    def update_mutated_names(self, renames: dict[str, str]) -> None:
        # [FUNCTION DOC] Records mutation_renames for the buffer names this node reads/writes that appear in renames, and
        # replaces its read/write dependencies with the renamed ones.
        self.mutation_renames = {
            name: renames[name]
            for name in (dep.name for dep in self.read_writes.reads_and_writes())
            if name in renames
        }
        self.set_read_writes(self.read_writes.rename(self.mutation_renames))

    def add_fake_dep(self, dep: Dep) -> None:
        # [FUNCTION DOC] Adds a (fake) read dependency dep to this node's read/writes.
        self.set_read_writes(self.read_writes.with_read(dep))

    def has_aliasing_or_mutation(self) -> bool:
        # [FUNCTION DOC] True if any output buffer aliases an input or mutates a buffer.
        return any(
            buf.get_aliases() or buf.get_mutations() for buf in self.get_outputs()
        )

    def set_read_writes(self, rw: dependencies.ReadWrites) -> None:
        # [FUNCTION DOC] Sets read/writes, resets unmet dependencies to all reads, clears caches depending on read/writes, and
        # prunes dependencies on already available buffers.
        self.read_writes = rw
        self.unmet_dependencies = self.read_writes.reads
        self.clear_read_writes_dependent_caches()
        self.prune_deps()

    def clear_read_writes_dependent_caches(self) -> None:
        # [FUNCTION DOC] Clears the cached coalesce analysis and tiling results of this node.
        self.get_coalesce_analysis.clear_cache(self)
        typing.cast(Any, self.get_tiling).clear_cache(self)

    @cache_on_self
    def get_coalesce_analysis(self) -> CoalesceVarAnalysis | None:
        # [FUNCTION DOC] Cached. Returns the memory coalescing analysis for a SchedulerNode or FusedSchedulerNode, else None.
        from .tiling_utils import _analyze_memory_coalescing

        if not isinstance(self, (SchedulerNode, FusedSchedulerNode)):
            return None
        return _analyze_memory_coalescing(self)

    @cache_on_self_and_args("BaseSchedulerNode")
    def get_tiling(
        self, numel: sympy.Expr, rnumel: sympy.Expr
    ) -> dict[str, sympy.Expr]:
        # [FUNCTION DOC] Cached per (numel, rnumel). Selects the SIMD tiling for this node's sub-nodes.
        from .codegen.simd import SIMDScheduling

        return SIMDScheduling.select_tiling(self.get_nodes(), numel, rnumel)

    def set_last_usage(
        self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]
    ) -> None:
        # [FUNCTION DOC] Computes last_usage: the buffers used (or aliased) by this node, mapped to their real names through
        # mutation_real_name, minus those still needed by future nodes (future_used_buffers).
        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = OrderedSet(mutation_real_name.get(k, k) for k in used_buffers)
        self.last_usage = used_buffers - future_used_buffers

    def mark_run(self) -> None:
        # [FUNCTION DOC] Marks the node as run by emitting allocation for each output buffer (codegen side effect).
        for buf in self.outputs:
            buf.allocate()

    def used_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Returns the names of all buffers read or written by this node.
        return OrderedSet(
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        )

    def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Returns the names of buffers read or written (excluding fake WeakDeps) plus, transitively, the buffers
        # they alias as outputs (alias names from V.graph).
        """
        Returns buffer names used by this node, including aliases.

        Note: is_fake WeakDeps are excluded since they are purely for ordering
        and should not affect buffer lifetime.
        """
        used_names: OrderedSet[str] = OrderedSet()

        deps = [
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
            if not (isinstance(dep, WeakDep) and dep.is_fake)
        ]
        while len(deps) > 0:
            dep = deps.pop()
            used_names.add(dep)
            if V.graph.name_to_buffer.get(dep):
                deps.extend(
                    alias
                    for alias in V.graph.name_to_buffer[
                        dep
                    ].get_inputs_that_alias_output()
                    if alias not in used_names
                )
        return used_names

    def prune_deps(self) -> None:
        # [FUNCTION DOC] Drops unmet dependencies whose buffers are already in the scheduler's available buffer names.
        self.unmet_dependencies = OrderedSet(
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        )

    def prune_weak_deps(self) -> None:
        # [FUNCTION DOC] Removes WeakDep reads whose defining operation was removed from the graph, updating read/writes.
        # Prune weak dependencies on operations that have been removed
        def should_prune(dep: Dep) -> bool:
            # [FUNCTION DOC] HELPER: True if dep is a WeakDep on a known buffer whose defining op was removed from the graph.
            if not isinstance(dep, WeakDep):
                return False
            if dep.name not in self.scheduler.name_to_buf:
                return False

            op_name = self.scheduler.name_to_buf[dep.name].defining_op_name()
            return op_name in V.graph.removed_operations

        to_remove = OrderedSet(
            dep for dep in self.read_writes.reads if should_prune(dep)
        )
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(
        self, name_to_fused_node: dict[str, BaseSchedulerNode]
    ) -> None:
        # [FUNCTION DOC] Prunes redundant dependencies using the module level prune helper with the fused-node map.
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the operation name of the wrapped IR node; raises AssertionError if node is None.
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_operation_name()

    def get_first_name(self) -> str:
        # [FUNCTION DOC] Returns the name of the first operation of this node (base: same as get_name).
        return self.get_name()

    @cache_on_self
    def get_operation_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Cached. Returns the set of names of all sub-nodes (operations).
        return OrderedSet(node.get_name() for node in self.get_nodes())

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Cached. Returns the set of names of this node's output buffers.
        return OrderedSet(out.get_name() for out in self.outputs)

    @cache_on_self
    def can_codegen_in_low_precision(self) -> bool:
        # [FUNCTION DOC] Cached. True if all sub-nodes are SchedulerNodes that can be codegen'd in low precision (disallowing fp32 ops).
        return all(
            isinstance(n, SchedulerNode)
            and can_codegen_without_upcasts(n, disallow_fp32_ops=True)
            for n in self.get_nodes()
        )

    @cache_on_self
    def can_codegen_without_upcasts(self) -> bool:
        # [FUNCTION DOC] Cached. True if all sub-nodes are SchedulerNodes that can be codegen'd without upcasts.
        return all(
            isinstance(n, SchedulerNode) and can_codegen_without_upcasts(n)
            for n in self.get_nodes()
        )

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        # [FUNCTION DOC] Returns the list of underlying nodes; base returns just this node.
        return [self]

    @cache_on_self
    def has_strict_sum(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node is a reduction ComputedBuffer with a strict-sum reduction block set.
        return any(
            isinstance(node, SchedulerNode)
            and isinstance(node.node, ComputedBuffer)
            and isinstance(node.node.data, ir.Reduction)
            and node.node.data.strict_sum_rblock is not None
            for node in self.get_nodes()
        )

    def get_outputs(self) -> Sequence[SchedulerBuffer]:
        # [FUNCTION DOC] Returns the output SchedulerBuffers.
        return self.outputs

    def get_output(self, buf_name: str) -> SchedulerBuffer:
        # [FUNCTION DOC] Returns the output SchedulerBuffer named buf_name (KeyError if absent).
        return self.outputs_by_name[buf_name]

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Returns the device of the wrapped IR node; raises AssertionError if node is None.
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return self.node.get_device()

    def is_cpu(self) -> bool:
        # [FUNCTION DOC] True if the node's device is a CPU device.
        device = self.get_device()
        return device is not None and device.type == "cpu"

    def is_gpu(self) -> bool:
        # [FUNCTION DOC] True if the node's device is a GPU type.
        device = self.get_device()
        return device is not None and is_gpu(device.type)

    def is_reduction(self) -> bool:
        # [FUNCTION DOC] Predicate: base returns False (overridden for reductions).
        return False

    def is_native_matmul(self) -> bool:
        # [FUNCTION DOC] Predicate: base returns False (native matmul template nodes override).
        return False

    def is_split_scan(self) -> bool:
        # [FUNCTION DOC] Predicate: base returns False (split scan nodes override).
        return False

    def is_template(self) -> bool:
        # [FUNCTION DOC] Predicate: base returns False (template nodes override).
        return False

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Predicate: base returns False (extern kernel nodes override).
        return False

    def is_foreach(self) -> bool:
        # [FUNCTION DOC] Predicate: base returns False (foreach nodes override).
        return False

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        # [FUNCTION DOC] Whether the buffer read via read_dep can be updated in place by this node; base returns False.
        return False

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Whether the node has side effects; base returns False.
        return False

    def decide_inplace_update(self) -> None:
        # [FUNCTION DOC] For SchedulerNodes with inplace buffers enabled, an active real kernel and device support, decides for each
        # output whether it can reuse an input buffer in place. Conditions: output allocatable, no aliases or mutations,
        # not removed, not comm buffer; the input is reusable, not defined by a no-op node, has this node as its only
        # remaining user that allows inplace, no cross-stream or cross-mempool hazard, compatible layout and size, and
        # all accesses in the fused node use a single index. On success records the inplace pair in the kernel args
        # and kernel inplace_update_buffers (and mutations for SIMD kernels). Mutates kernel state; returns None.
        """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
        from .codegen.wrapper import can_match_buffer_size

        if not (
            isinstance(self, SchedulerNode)
            and config.inplace_buffers
            and V.graph.has_feature(self.get_device(), BackendFeature.INPLACE_BUFFERS)
            and (
                not isinstance(V.kernel, torch._inductor.codegen.simd.SIMDKernel)
                or getattr(V.kernel, "mutations", None) is not None
            )
            # hacky check for if V.kernel is a real kernel or NullHandler
            and hasattr(V.kernel, "args")
        ):
            return

        # NOTE remove V.graph.removed_operations once deps issue is fixed
        inconsequential_nodes = (
            self.ancestors
            | V.graph.removed_operations
            | self.scheduler.completed_operations
        )

        def single_index_in_fused_node(buf_to_be_inplaced: SchedulerBuffer) -> bool:
            # [FUNCTION DOC] HELPER: True if, among users of the buffer inside the same fused node as this node, all reads/writes of
            # the buffer use a single equivalent access (deduplicated dependency set has at most one element).
            # Inside of NodeUser, we track that the read and write are equivalent
            # before deciding if the use can be inplace.
            # But if that use is fused into a larger kernel, we need to check equivalence
            # of other accesses in fused scheduler node as well.
            fused_node = buf_to_be_inplaced.scheduler.get_fused_node(self)
            buf_name = buf_to_be_inplaced.get_name()
            # Dedup read/writes with equivalent indices
            # TODO - would be nice if we could just cache accesses on ReadWrites,
            # and enforce variant that this class & members are functional..
            deps: OrderedSet[Dep] = OrderedSet()
            for user in buf_to_be_inplaced.users:
                user_node = user.node
                if not isinstance(user_node, BaseSchedulerNode):
                    continue

                if (
                    user_node.get_first_name()
                    not in buf_to_be_inplaced.scheduler.name_to_fused_node
                    or buf_to_be_inplaced.scheduler.get_fused_node(user_node)
                    is not fused_node
                ):
                    continue

                deps |= (
                    o
                    for o in user_node.read_writes.reads_and_writes()
                    if o.name == buf_name
                )
                if len(deps) > 1:
                    return False

            return True

        for buf in self.get_outputs():
            buf_node = buf.node
            if buf_node is None:
                raise AssertionError("expected buf_node to be set")
            if (
                not buf_node.should_allocate()
                or buf_node.get_inputs_that_alias_output()
                or buf_node.get_mutation_names()
                or buf.get_name() in V.graph.removed_buffers
                # CommBufferLayout buffer must keep its P2P allocation.
                # Do not allow in-place reuse into or from a P2P buffer.
                or isinstance(buf_node.get_output_spec(), ir.CommBufferLayout)
            ):
                continue

            for read in self.read_writes.reads:
                input_buf: SchedulerBuffer | SchedulerDonatedBuffer | None
                if read.name in self.scheduler.name_to_donated_buffer:
                    input_buf = self.scheduler.name_to_donated_buffer[read.name]
                else:
                    input_buf = self.scheduler.name_to_buf.get(read.name)

                if (
                    input_buf
                    and V.graph.wrapper_code.can_reuse(input_buf, self)
                    and not isinstance(input_buf.defining_op, NopKernelSchedulerNode)
                ):
                    if input_buf.users is None:
                        raise AssertionError("expected input_buf.users to be set")
                    remaining_uses = [
                        x
                        for x in input_buf.users
                        if x.node.get_name() not in inconsequential_nodes
                    ]
                    has_cross_stream_hazard = self.scheduler.has_cross_stream_hazard(
                        read.name, self
                    )
                    has_cross_mempool_hazard = self.scheduler.get_buf_mempool(
                        read.name
                    ) != self.scheduler.node_to_mempool.get(self)

                    if (
                        not has_cross_stream_hazard
                        and not has_cross_mempool_hazard
                        and len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and input_buf.node is not None
                        and not isinstance(
                            input_buf.node.get_output_spec(),
                            (
                                ir.NoneLayout,
                                ir.MultiOutputLayout,
                                ir.MutationLayoutSHOULDREMOVE,
                                ir.CommBufferLayout,
                            ),
                        )
                        and not (
                            input_buf.defining_op
                            and isinstance(
                                input_buf.defining_op.node,
                                (ir.FallbackKernel, ir.MultiOutput),
                            )
                            and len(input_buf.node.get_inputs_that_alias_output()) > 0
                        )
                        and can_match_buffer_size(input_buf.node, buf.node)
                        and single_index_in_fused_node(input_buf)
                    ):
                        # if there isn't a triton kernel, then we don't need to call triton-specific things.
                        # but TODO this might be a convenient place to signal to the Collective kernels to inplace
                        # (and, can we make "kernel" less generic of a name?)
                        V.kernel.args.make_inplace(input_buf.get_name(), buf.get_name())
                        # mutations not tracked in cpp kernels
                        if isinstance(
                            V.kernel, torch._inductor.codegen.simd.SIMDKernel
                        ):
                            V.kernel.mutations.add(input_buf.get_name())
                            V.kernel.mutations.add(buf.get_name())

                        V.kernel.inplace_update_buffers[buf.get_name()] = (
                            input_buf.get_name()
                        )
                        break

    def codegen_originating_info(
        self, buffer: IndentedBuffer, only_once: bool = True
    ) -> None:
        # [FUNCTION DOC] When config.comment_origin is set and (if only_once) not yet written, writes pragma comment lines
        # describing the originating fx nodes (op, target, seq_nr, last stack trace line) of the IR node to buffer, then
        # marks written. Skips output nodes. Raises AssertionError if node is None.
        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        origins = self.node.get_origins()
        out_lines = []

        for o in origins:
            if o.op == "output":
                # These are boring and samey
                continue

            out_lines.append("")
            # TODO(voz): Should the pragma be constant somewhere?
            out_lines.append("#pragma CMT ORIGIN:")
            op_info_str = f"#pragma CMT {o.op} {o.target}"
            if "seq_nr" in o.meta:
                op_info_str = op_info_str + f" seq_nr:{o.meta['seq_nr']}"
            out_lines.append(op_info_str)
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.rsplit("|", maxsplit=1)[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                    .replace(
                        "\\", "\\\\"
                    )  # For windows safe path, avoid for example \x, \U.
                )
                out_lines.append("#pragma CMT END ORIGIN")
                out_lines.append("")

        if len(out_lines) == 0:
            return

        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        buffer.writelines(out_lines)
        self.written = True

    @cache_on_self
    def get_read_write_buffers_sizes(self) -> int:
        # [FUNCTION DOC] Cached. Total bytes read plus written by this node.
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=True
        )

    @cache_on_self
    def get_read_buffer_sizes(self) -> int:
        # [FUNCTION DOC] Cached. Total bytes read by this node.
        return self.get_read_write_buffers_sizes_impl(
            include_reads=True, include_writes=False
        )

    @cache_on_self
    def get_write_buffer_sizes(self) -> int:
        # [FUNCTION DOC] Cached. Total bytes written by this node.
        return self.get_read_write_buffers_sizes_impl(
            include_reads=False, include_writes=True
        )

    def get_read_write_buffers_sizes_impl(
        self, include_reads: bool, include_writes: bool
    ) -> int:
        # [FUNCTION DOC] Sums the per-buffer byte accesses (reads and/or writes as selected) of this node.
        return sum(
            self.get_read_write_buffer_accesses(
                include_reads=include_reads, include_writes=include_writes
            ).values(),
            start=0,
        )

    def get_read_write_buffer_accesses(
        self, include_reads: bool, include_writes: bool
    ) -> dict[str, int]:
        # [FUNCTION DOC] Returns a mapping from buffer name to estimated bytes accessed by this node (reads and/or writes as selected).
        # Per buffer, accessed elements = node numel (1e9 for non-SchedulerNodes) times number of deps, capped by the buffer
        # element count, times dtype size. Empty for no-op nodes, MultiOutput extern nodes and graphsafe rng fallbacks.
        # Writes to buffers fully consumed inside a fused node are dropped. MultiOutputLayout buffers sum their MultiOutput
        # users (0 if other users exist); none-layout buffers sum the mutated buffers' bytes. Names not in the graph
        # buffers or inputs are skipped.
        """
        Counting the number of bytes accessed for a kernel is
        surprisingly tricky. In particular, there is a differentiation
        between 'theoretical' memory accesses and practical memory
        accesses. For example, a layernorm kernel may actually access an
        input 3 times, but in theory, it only needs to access its input
        once (and may be optimized to do so through say, persistent
        reductions)

        Another example is that even though a buffer is passed in, we may
        not access the entire buffer. This may occur if we are accessing
        a slice of the buffer. Another tricky case is for indirect
        indexing, where the amount of bytes accessed depends on the
        values of the input.

        What this function aims to compute is the memory accesses for
        worst-case inputs, best-case optimization. What this means is
        that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

        1. Numel in ranges multiplied by number of deps the buffer has
        2. The buffer size

        Returns memory accesses per buffer.
        """
        if isinstance(self, NopKernelSchedulerNode):
            return {}
        if isinstance(self, ExternKernelSchedulerNode) and isinstance(
            self.node, MultiOutput
        ):
            # todo: Calculate this - it's kinda annoying.
            return {}
        if (
            isinstance(self, ExternKernelSchedulerNode)
            and isinstance(self.node, ir.FallbackKernel)
            and self.node.op_overload
            is torch._prims.rng_prims.graphsafe_run_with_rng_state
        ):
            return {}

        def try_size_hint(s: sympy.Expr) -> int:
            # [FUNCTION DOC] HELPER: Returns the integer optimization hint of a size expression, with fallback 0.
            return V.graph.sizevars.optimization_hint(s, fallback=0)

        if isinstance(self, SchedulerNode):
            node_numel = try_size_hint(
                sympy_product(self.get_ranges()[0])
                * sympy_product(self.get_ranges()[1]),
            )
        else:
            node_numel = int(1e9)
        buf_accesses = collections.defaultdict(list)

        if include_reads:
            for dep in self.read_writes.reads:
                buf_accesses[dep.name].append(dep)

        if include_writes:
            for dep in self.read_writes.writes:
                buf_accesses[dep.name].append(dep)

        reads = (
            OrderedSet(dep.name for dep in self.read_writes.reads)
            if include_reads
            else OrderedSet()
        )
        writes = (
            OrderedSet(dep.name for dep in self.read_writes.writes)
            if include_writes
            else OrderedSet()
        )

        def is_materialized(buf: str, snodes: Sequence[BaseSchedulerNode]) -> bool:
            # [FUNCTION DOC] HELPER: True if buf is used by some node outside the given snodes (so it must be materialized).
            users = self.scheduler.name_to_buf[buf].users
            buf_uses = OrderedSet(user.node for user in users)
            return len(buf_uses - OrderedSet(snodes)) > 0

        if isinstance(self, FusedSchedulerNode):
            removed_buffers = OrderedSet(
                dep for dep in writes if not is_materialized(dep, self.snodes)
            )
            writes = writes - removed_buffers
            reads = reads - removed_buffers

        buf_byte_accesses: dict[str, int] = {}

        for buf_name in reads | writes:
            buf_accessed_elems = sum(node_numel for dep in buf_accesses[buf_name])
            buf: ir.Buffer | ir.TensorBox | ir.TorchBindObject
            if buf_name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf_name]
            elif buf_name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf_name]
            else:
                continue

            def get_buf_bytes(
                buf: ir.Buffer | ir.TensorBox | ir.TorchBindObject | None,
            ) -> int:
                # [FUNCTION DOC] HELPER: Recursive byte count for a buffer: 0 if falsy; TorchBind object bytes; sum over MultiOutput users for
                # MultiOutputLayout (raises AssertionError if a user is not a scheduler node); mutated buffers for NoneLayout;
                # else dtype size times min(accessed elements, buffer elements).
                if not buf:
                    return 0

                if isinstance(buf, ir.TorchBindObject):
                    return buf.get_buf_bytes()
                elif isinstance(buf.layout, MultiOutputLayout):
                    # Kind of a lazy way to get the MultiOutput nodes corresponding to
                    # a MultiOutputLayout
                    users = self.scheduler.name_to_buf[buf.get_name()].users
                    tot = 0
                    for user in users:
                        if isinstance(user.node, OutputNode):
                            continue
                        if not isinstance(user.node, BaseSchedulerNode):
                            raise AssertionError(
                                "expected user.node to be a BaseSchedulerNode"
                            )
                        if isinstance(user.node.node, MultiOutput):
                            for sched_buf in user.node.get_outputs():
                                tot += get_buf_bytes(sched_buf.node)
                        else:
                            # Buf is a MultiOutputLayout but not all of its
                            # users are MultiOutputs...
                            # TODO: Figure out what's going on
                            return 0
                    return tot
                elif isinstance(buf.layout, ir.NoneLayout):
                    return sum(
                        get_buf_bytes(V.graph.get_buffer(mut_name))
                        for mut_name in buf.get_mutation_names()
                    )
                else:
                    buf_elems = try_size_hint(sympy_product(buf.get_size()))
                    return get_dtype_size(buf.get_dtype()) * min(
                        buf_accessed_elems, buf_elems
                    )

            buf_bytes = get_buf_bytes(buf)
            if buf_name not in buf_byte_accesses:
                buf_byte_accesses[buf_name] = buf_bytes
            else:
                buf_byte_accesses[buf_name] += buf_bytes

        return buf_byte_accesses

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # [FUNCTION DOC] Cached. Estimates FLOPs for this node from its originating fx node (None if unavailable). Resolves symbolic
        # counts to an int hint (fallback 0) and adds it to the inductor flop_count counter (side effect).
        if self.node is None:
            return None
        fx_node = self.node.get_origin_node()
        if fx_node is None:
            return None

        flops = count_flops_fx(fx_node)
        if flops is None:
            return None

        if isinstance(flops, torch.SymInt):
            flops = flops.node.expr

        resolved_flops = V.graph.sizevars.optimization_hint(flops, fallback=0)
        counters["inductor"]["flop_count"] += resolved_flops
        return resolved_flops

    def get_estimated_runtime(self) -> float:
        # [FUNCTION DOC] Returns the estimated runtime in ms: override_estimated_runtime when set, else the cached estimate.
        if self.override_estimated_runtime is not None:
            return self.override_estimated_runtime

        return self._get_estimated_runtime()

    @cache_on_self
    def _get_estimated_runtime(self) -> float:
        # [FUNCTION DOC] Cached. Returns estimated runtime in milliseconds. 0 for non-GPU outputs and wait nodes. Collectives use an
        # NCCL library estimator (with a local cache and fallback to the in-tree algorithmic estimate) or just the
        # algorithmic one per config; unestimable collectives (ValueError/TypeError) give 0. Otherwise tries a benchmark
        # estimate for supported extern ops; else the roofline max of compute time (FLOPs over device TFLOPs) and memory
        # time (bytes over DRAM bandwidth), memory-only if no FLOPs; 0 if device properties are unusable.
        """
        Returns estimated op runtime in milliseconds (ms)
        """
        buf = self.get_nodes()[0].get_outputs()[0]
        layout = buf.node.get_output_spec()
        if not is_gpu(get_device_type(layout)):
            # default to no reordering based on runtime
            return 0

        # Collective kernels
        if is_collective(self.node):
            if not isinstance(self.node, ir.IRNode):
                raise AssertionError("expected self.node to be an ir.IRNode")
            try:
                if config_comms.runtime_estimations_use_nccl_lib_estimations:
                    cache_key = get_estimate_runtime_cache_key_from_snode(self)
                    cache = get_estimate_runtime_cache()
                    cache_val = cache.lookup(cache_key)
                    if cache_val is not None:
                        if not isinstance(cache_val, float):
                            raise AssertionError("expected cache_val to be a float")
                        return cache_val

                    ms = estimate_nccl_collective_runtime_nccl_estimator(self)
                    if ms is None:
                        # NCCL estimations fail: fallback to in-tree algorithmic estimation.
                        ms = estimate_nccl_collective_runtime(self.node)

                    cache.set_value(cache_key, value=ms)
                    return ms
                return estimate_nccl_collective_runtime(self.node)
            except ValueError as e:
                # We don't know how to estimate runtime for this collective,
                # falling back to 0
                log.info(e)
                return 0
            except TypeError as e:
                # this happens when the collective is not of type ir._CollectiveKernel
                log.info(e)
                return 0

        elif is_wait(self.node):
            # ir.Wait is only used for collective ops.
            # The time needed for the collective op is already estimated and considered
            # when we are processing the collective op IR node, so ir.Wait takes 0 time
            # since it doesn't take extra time to get the result after the collective is completed.
            return 0

        ret = maybe_estimate_runtime_benchmark(self)
        if ret is not None:
            return ret

        dtype = buf.node.maybe_get_dtype()
        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10**12
            # If cudaGetDeviceProperties returns 0 for gpu_memory_bandwidth or gpu_flops
            # there is a chance to continue execution successfully. Otherwise, it would fail with
            # ZeroDivisionError below.
            if gpu_memory_bandwidth <= 0:
                raise AssertionError(
                    f"gpu_memory_bandwidth cannot be <= 0, but got {gpu_memory_bandwidth}"
                )
            if gpu_flops <= 0:
                raise AssertionError(f"gpu_flops cannot be <= 0, but got {gpu_flops}")
        except Exception:
            return 0

        flops_est = self.estimate_flops()

        if flops_est == 0 or flops_est is None:
            # no flops estimate, so fall back to memory estimate
            ns = self.get_read_write_buffers_sizes() / gpu_memory_bandwidth
            ms = ns / 1e6
            return ms

        # TODO(xmfan): find a better heuristic to model FLOPS/latency relationship
        factor = 1.0
        counted_bytes = self.get_read_write_buffers_sizes()
        counted_bytes = 0 if counted_bytes is None else counted_bytes
        compute_time = (factor * flops_est / gpu_flops) * 1e9
        transfer_time = counted_bytes / gpu_memory_bandwidth

        # Return estimated runtime in milliseconds
        ns = max(compute_time, transfer_time)
        ms = ns / 1e6
        return ms

    def get_template_node(self) -> ir.TemplateBuffer | None:
        # [FUNCTION DOC] Returns the template buffer if this node is a template; base returns None.
        return None

    def get_template_node_or_throw(self) -> ir.TemplateBuffer:
        # [FUNCTION DOC] Returns the template buffer; raises AssertionError if there is none.
        template = self.get_template_node()
        if template is None:
            raise AssertionError("expected template to be set")
        return template

    @staticmethod
    def get_prologue_template_epilogue(
        nodes: list[BaseSchedulerNode],
    ) -> tuple[list[BaseSchedulerNode], BaseSchedulerNode, list[BaseSchedulerNode]]:
        # [FUNCTION DOC] Static. Splits a node list around its first template node into (prologue nodes, template node,
        # epilogue nodes). Raises StopIteration if no template node exists.
        """
        For the list of nodes, get the prologue, template, and epilogue
        """
        template_index = next(i for i, n in enumerate(nodes) if n.is_template())

        prologue = nodes[:template_index]
        template_node = nodes[template_index]
        epilogue = nodes[template_index + 1 :]
        return prologue, template_node, epilogue


@functools.cache
def get_estimate_runtime_cache() -> torch._inductor.codecache.LocalCache:
    # [FUNCTION DOC] HELPER: Cached (process-wide). Returns the LocalCache used to persist runtime estimates.
    return torch._inductor.codecache.LocalCache()


def get_estimate_runtime_cache_key_from_snode(snode: BaseSchedulerNode) -> str:
    # [FUNCTION DOC] HELPER: Builds a string cache key for a collective/extern node from its python kernel name and the sizes of its
    # tensor IR arguments (non-tensor arguments as None), after filling default arguments.
    python_kernel_name = getattr(snode.node, "python_kernel_name", "")
    args = snode.node.inputs  # type: ignore[union-attr]
    args = snode.node.fill_non_provided_args(  # type: ignore[union-attr]
        [*args, *snode.node.constant_args],  # type: ignore[union-attr]
        snode.node.kwargs,  # type: ignore[union-attr]
    )
    kwargs = snode.node.kwargs  # type: ignore[union-attr]
    flat_args, flat_args_pytree_spec = pytree.tree_flatten((args, kwargs))

    def _is_tensor_ir(x) -> bool:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: True if x is an IR node other than generator or opaque object state.
        return isinstance(x, ir.IRNode) and not isinstance(
            x, (ir.GeneratorState, ir.OpaqueObjectState)
        )

    cache_key = str(
        (python_kernel_name,)
        + tuple(tuple(a.get_size()) if _is_tensor_ir(a) else None for a in flat_args)
    )
    return cache_key


def _get_benchmarkable_extern_fn(
    snode: BaseSchedulerNode,
) -> Callable[[Any], Any] | None:
    # [FUNCTION DOC] HELPER: Returns the op overload packet for an extern kernel node when it has an OpOverload registered in the
    # flop counter registry, else None (non-extern nodes and unregistered ops give None).
    if not isinstance(snode, ExternKernelSchedulerNode):
        return None
    if not isinstance(snode.node, ir.ExternKernel):
        return None

    op_overload = snode.node.op_overload
    if not isinstance(op_overload, torch._ops.OpOverload):
        return None

    op = op_overload.overloadpacket

    from torch.utils.flop_counter import flop_registry

    if op not in flop_registry:
        return None

    return op


def maybe_estimate_runtime_benchmark(snode: BaseSchedulerNode) -> float | None:
    # [FUNCTION DOC] Benchmarks a supported extern op to estimate runtime in ms when config.runtime_estimations_mms_benchmark is set;
    # returns None otherwise or if not benchmarkable. Uses the shared estimate cache (raises AssertionError on a
    # non-float cached value) and stores new measurements in it (side effect: runs the op on the device).
    bench_fn = None
    args_kwargs_fn = None
    if config.runtime_estimations_mms_benchmark:
        mm_fn = _get_benchmarkable_extern_fn(snode)
        if mm_fn is None:
            return None
        bench_fn = mm_fn

        args_kwargs_fn = lambda: snode_args_kwargs(snode)  # noqa: E731
    else:
        return None

    cache_key = get_estimate_runtime_cache_key_from_snode(snode)
    cache = get_estimate_runtime_cache()
    cache_val = cache.lookup(cache_key)
    if cache_val is not None:
        if not isinstance(cache_val, float):
            raise AssertionError("expected cache_val to be a float")
        return cache_val

    from .utils import snode_args_kwargs

    args, kwargs = args_kwargs_fn()
    from torch._inductor.runtime.benchmarking import benchmarker

    ms = benchmarker.benchmark(
        bench_fn,
        args,  # pyrefly: ignore[bad-argument-type]
        kwargs,
        memory_warmup_iters=5,
        benchmark_iters=10,
        max_benchmark_duration=10,
    )  # type: ignore[arg-type]

    cache.set_value(cache_key, value=ms)
    return ms


@dataclasses.dataclass(slots=True)
class WhyNoFuse:
    # [CLASS DOC] Records why two nodes were not fused. Constructed with the two nodes (stores names); calling it with a
    # format string and arguments stores the reason and logs it to the fusion log. String conversion formats the
    # message.
    # KNOWN ISSUE: reason and args are only set when the instance is called, so converting to a string before any call
    # raises AttributeError (slots, no defaults).
    name1: str
    name2: str
    reason: str
    args: tuple[Any, ...]

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] Stores the names of the two nodes.
        self.name1 = node1.get_name()
        self.name2 = node2.get_name()

    def __call__(self, reason: str, *args: Any) -> None:
        # [FUNCTION DOC] Stores the reason format string and its arguments and logs the object at debug level to the fusion log.
        self.reason = reason
        self.args = args
        fusion_log.debug(self)

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the message that the two named nodes cannot be fused, with the reason formatted using its arguments.
        return f"cannot fuse {self.name1} with {self.name2}: " + (
            self.reason % self.args
        )


def pformat(obj: Any) -> str:
    # [FUNCTION DOC] Pretty-formats obj for debug output (sets sorted by string form); multi-line results are indented and start on a new line.
    if isinstance(obj, (OrderedSet, set)):  # noqa: set_linter
        # pformat has trouble with sets of sympy exprs
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' ' * 4)}"
    return result


class OutputNode:
    # [CLASS DOC] Sentinel pseudo-node standing for the graph output as a user of buffers; carries one unmet StarDep and
    # has name OUTPUT, no aliasing and is not a reduction.
    def __init__(self, dep: StarDep) -> None:
        # [FUNCTION DOC] Stores the single StarDep as the only unmet dependency.
        self.unmet_dependencies = OrderedSet([dep])

    def is_reduction(self) -> bool:
        # [FUNCTION DOC] Always False.
        return False

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        # [FUNCTION DOC] Returns an empty tuple (no aliasing).
        return ()

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the constant name OUTPUT.
        return "OUTPUT"

    __repr__ = get_name


def _prune_redundant_deps(
    node: BaseSchedulerNode,
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
) -> None:
    # [FUNCTION DOC] HELPER: Removes redundant WeakDep (mutation ordering) unmet dependencies of node: those whose fused upstream
    # node is already a real dependency and fusable, or that point to the node itself. Updates unmet dependencies
    # and read/writes in place (side effect). Enforces that fusions happen in order.
    """
    Prunes weakdeps intended for mutation ordering
    on an upstream fused node if after fusion there is another dependency
    on the fused upstream node, making the weakdep redundant

    In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
    be incrementally removed, enabling other fusions, ensuring they are fused in order.
    """
    name_to_dep_count: Counter[str] = collections.Counter()

    for dep in node.unmet_dependencies:
        if not isinstance(dep, WeakDep):
            op_name = name_to_buf[dep.name].defining_op_name()
            name_to_dep_count[name_to_fused_node[op_name].get_name()] += 1

    def should_prune(dep: Dep) -> bool:
        # [FUNCTION DOC] HELPER: True if dep is a redundant or self-referential WeakDep as defined by the enclosing function.
        if isinstance(dep, WeakDep):
            op_name = name_to_buf[dep.name].defining_op_name()
            is_redundant = name_to_dep_count[
                name_to_fused_node[op_name].get_name()
            ] > 0 and node.scheduler.fusable_weak_dep(
                dep, name_to_fused_node[op_name], node
            )
            # These can occur because fused nodes always gather deps from their snodes
            # If B has a weakdep on A
            # B gets fused with C, then any time BC is fused, the weakdep will reappear
            is_self_dep = name_to_fused_node[op_name] == node
            return is_redundant or is_self_dep
        else:
            return False

    deps_to_prune = OrderedSet(
        dep for dep in node.unmet_dependencies if should_prune(dep)
    )

    if deps_to_prune:
        node.unmet_dependencies = node.unmet_dependencies - deps_to_prune
        node.set_read_writes(node.read_writes.remove_reads(deps_to_prune))


class ExternKernelSchedulerNode(BaseSchedulerNode):
    # [CLASS DOC] Scheduler node wrapping an extern (fallback) kernel. Reports itself as extern, may report side effects, and
    # for user-defined Triton kernels that allow epilogue fusion exposes a group of mutated-arg numel with rnumel 1.
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None:
        # [FUNCTION DOC] Builds the node from the IR operation, sets its read/writes, and for a user-defined Triton kernel that can fuse
        # an epilogue sets group to (device, (numel of first mutable arg, 1)).
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())

        if isinstance(node, ir.UserDefinedTritonKernel) and node.can_fuse_epilogue():
            numel = math.prod(node.mutable_args[0].shape)
            rnumel = 1
            device = node.get_device_or_error()
            # pyrefly: ignore [bad-assignment]
            self.group = (device, (numel, rnumel))

    def debug_str_extra(self) -> str:
        # [FUNCTION DOC] Returns an extra debug line with the node name and python kernel name.
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'python_kernel_name', None)}"

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] True if the wrapped IR node reports side effects; raises AssertionError if node is None.
        if self.node is None:
            raise AssertionError("expected self.node to be set")
        return hasattr(self.node, "has_side_effects") and self.node.has_side_effects()

    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]:
        # [FUNCTION DOC] Returns (iteration ranges, reduction ranges): one range of the first mutable arg numel for epilogue-fusable
        # user Triton kernels, otherwise two empty lists.
        if (
            isinstance(self.node, ir.UserDefinedTritonKernel)
            and self.node.can_fuse_epilogue()
        ):
            numel = math.prod(self.node.mutable_args[0].shape)
            return ([numel], [])
        return ([], [])

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Delegates code generation to the wrapped ExternKernel; raises AssertionError if the node is not an ExternKernel.
        if not isinstance(self.node, ir.ExternKernel):
            raise AssertionError("expected self.node to be an ir.ExternKernel")
        return self.node.codegen(wrapper)


class NopKernelSchedulerNode(BaseSchedulerNode):
    # [CLASS DOC] Scheduler node for no-op IR operations (no kernel is generated); only records read/writes.
    def __init__(self, scheduler: Scheduler, node: ir.Operation) -> None:
        # [FUNCTION DOC] Builds the node from the IR operation and sets its read/writes.
        super().__init__(scheduler)
        self._init_from_node(node)
        self.set_read_writes(node.get_read_writes())


class SchedulerNode(BaseSchedulerNode):
    # [CLASS DOC] Node wrapping a ComputedBuffer or TemplateBuffer. State: loop body (_body), iteration sizes (_sizes), group
    # (device, iteration sizes), an optional loop mutation listener and a loop-state generation counter bumped on every loop
    # transformation (used for caching across fusion search). Provides loop reordering/reindexing/merging
    # transformations that update the body, sizes, group and dependencies, plus snapshot/restore of that state.
    """
    A SchedulerNode is a node for scheduling that encapsulates either
    a ComputedBuffer or a TemplateBuffer.
    """

    _sizes: tuple[Sequence[sympy.Expr], ...]
    _body: LoopBody

    def __init__(
        self,
        scheduler: Scheduler,
        node: ir.ComputedBuffer | ir.TemplateBuffer,
    ) -> None:
        # [FUNCTION DOC] Initializes base state, no loop mutation listener, generation 0, then reads the IR node and computes loop
        # attributes.
        super().__init__(scheduler)
        self._loop_mutation_listener: Callable[[SchedulerNode], None] | None = None
        self._loop_state_gen = 0
        self._init_from_node(node)
        self._compute_attrs()

    def _compute_attrs(
        self,
        extra_indexing_constraints: ir.ExtraIndexingConstraints | None = None,
        recompute_sizes_body_func: Callable[_P, _T] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Recomputes sizes and body by simplifying/reordering the IR node (optionally with extra indexing
        # constraints or a size-body recompute function), sets group using the backend group function, and
        # sets read/writes (normalized unless loop ordering after fusion is enabled on GPU). Raises AssertionError for
        # unsupported node types.
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(
                "expected self.node to be a ComputedBuffer or TemplateBuffer"
            )
        self._sizes, body = self.node.simplify_and_reorder(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )
        self._body = body  # type: ignore[assignment]

        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Don't normalize since normalization will merge loops which
        # makes it hard to decide new loop orders.
        should_normalize = not config.loop_ordering_after_fusion or not is_gpu(
            device.type
        )

        if isinstance(self.node, ir.TemplateBuffer):
            self.set_read_writes(
                self.node.extract_read_writes(normalize=should_normalize)
            )
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=should_normalize
                )
            )

    def recompute_size_and_body(
        self,
        extra_indexing_constraints: ir.ExtraIndexingConstraints | None = None,
        recompute_sizes_body_func: Callable[..., Any] | None = None,
    ) -> None:
        # [FUNCTION DOC] Recomputes sizes and body, preserving manually added fake (Weak/Star) dependencies and mutation renames in
        # the new read/writes.
        fake_deps: OrderedSet[Dep] = OrderedSet(
            dep for dep in self.read_writes.reads if isinstance(dep, (WeakDep, StarDep))
        )
        self._compute_attrs(
            extra_indexing_constraints=extra_indexing_constraints,
            recompute_sizes_body_func=recompute_sizes_body_func,
        )
        if fake_deps or self.mutation_renames:
            read_writes = self.read_writes
            if fake_deps:
                read_writes = read_writes.with_read(fake_deps)
            self.set_read_writes(read_writes.rename(self.mutation_renames))

    def refresh_dependencies(
        self, normalize: bool, need_clear_tiling_cache: bool
    ) -> None:
        # [FUNCTION DOC] Recomputes read/writes from the current loop body (optionally normalized), re-applying fake dependencies
        # and mutation renames, then clears dependent caches.
        # Fake dependencies are added manually. They can not be analyzed from
        # extract_read_writes. Find them out and apply manually.
        fake_deps: OrderedSet[Dep] = OrderedSet(
            dep for dep in self.read_writes.reads if isinstance(dep, (WeakDep, StarDep))
        )

        # don't normalize since the loop order may need to be further changed
        # later
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=normalize
            )
            .with_read(fake_deps)
            .rename(self.mutation_renames)
        )

        self.clear_loop_body_dependent_caches(need_clear_tiling_cache)

    def clear_loop_body_dependent_caches(self, need_clear_tiling_cache: bool) -> None:
        # [FUNCTION DOC] Clears read/writes caches and the pointwise read/writes cache; if requested also clears the global
        # SIMD candidate tilings cache (side effect).
        self.clear_read_writes_dependent_caches()
        self.pointwise_read_writes.clear_cache(self)

        if need_clear_tiling_cache:
            from .codegen.simd import SIMDScheduling

            # TODO(shunting) if this cause compilation time increase when
            # enabling LOAF by default, try just clearing the specific cache
            # entry by using a customized cache implementation rather than
            # lru_cache.
            SIMDScheduling.candidate_tilings.cache_clear()

    def snapshot_loop_state(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Returns a tuple snapshot of the mutable loop state (body, sizes, group, read/writes, unmet dependencies,
        # generation). Must stay in sync with the transformation methods and restore_loop_state.
        """Snapshot mutable state modified by loop transformations
        (apply_new_loop_order, apply_loop_reindexing). Must be kept
        in sync with those methods and restore_loop_state."""
        return (
            self._body,
            self._sizes,
            self.group,
            self.read_writes,
            self.unmet_dependencies,
            self._loop_state_gen,
        )

    def restore_loop_state(self, state: tuple[Any, ...]) -> None:
        # [FUNCTION DOC] Restores the loop state from a snapshot and clears dependent caches (including tiling).
        """Restore state from snapshot_loop_state."""
        (
            self._body,
            self._sizes,
            self.group,
            self.read_writes,
            self.unmet_dependencies,
            self._loop_state_gen,
        ) = state
        self.clear_loop_body_dependent_caches(need_clear_tiling_cache=True)

    def _before_loop_state_mutation(self) -> None:
        # [FUNCTION DOC] HELPER: Notifies the loop mutation listener (if any) before a change, then increments the loop-state generation.
        if self._loop_mutation_listener is not None:
            self._loop_mutation_listener(self)
        # Identifies the current loop state, so analyses derived from it can be
        # cached across the O(n^2) fusion pair search. Bumped after notifying
        # the listener, which snapshots the pre-mutation state: snapshot and
        # restore then carry the generation, so rolling a trial reindex back
        # also restores cache validity.
        self._loop_state_gen += 1

    def apply_indexing_exprs(self, replacements: dict[str, sympy.Expr]) -> None:
        # [FUNCTION DOC] Applies replacement indexing expressions to the loop body and refreshes dependencies (normalized).
        # Raises AssertionError if there is no body.
        if self._body is None:
            raise AssertionError("expected a loop body")
        self._before_loop_state_mutation()
        self._body = self._body.with_indexing_exprs(replacements)
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=True)

    def apply_new_loop_order(self, new_order: Sequence[int]) -> None:
        # [FUNCTION DOC] Reorders iteration loops of the body by new_order, updates sizes and refreshes dependencies (not normalized).
        self._before_loop_state_mutation()
        self._body = self._body.reorder_iter_loops(
            new_order,
        )
        self._sizes = self._body.sizes

        self.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

    def apply_loop_reindexing(self, new_iter_sizes: Sequence[sympy.Expr]) -> None:
        # [FUNCTION DOC] Reindexes iteration loops to new_iter_sizes, updates sizes and group, refreshes dependencies. Raises
        # AssertionError for unsupported node types.
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(
                "expected self.node to be a ComputedBuffer or TemplateBuffer"
            )

        self._before_loop_state_mutation()
        self._body = self._body.reindex_iter_loops(new_iter_sizes)
        self._sizes = self._body.sizes

        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        self.refresh_dependencies(normalize=False, need_clear_tiling_cache=True)

    def swap_pw_red_dimension(self) -> None:
        # [FUNCTION DOC] Swaps pointwise and reduction dimensions by reordering loops (reduction dims first) and swapping the two
        # entries of group. Raises AssertionError if group does not have exactly two parts.
        num_rdims = self._body.get_original_num_rdims()
        num_pwdims = len(self._body.iter_vars) - num_rdims
        pwdims = tuple(range(num_pwdims))
        rdims = tuple(range(num_pwdims, num_pwdims + num_rdims))

        self.apply_new_loop_order(rdims + pwdims)
        if len(self.group[1]) != 2:
            raise AssertionError(
                f"expected group[1] to have length 2, got {len(self.group[1])}"
            )
        self.group = self.group[0], (self.group[1][1], self.group[1][0])

    def extract_pw_from_reduction(self) -> BaseSchedulerNode:
        # [FUNCTION DOC] Replaces the body by its pointwise-only extraction from the reduction and returns self.
        self._body = self._body.extract_pw_from_reduction()
        return self

    def cancel_reduction_split(self) -> None:
        # [FUNCTION DOC] If this node is a split reduction, recomputes its attributes using the original (unsplit) inner function;
        # otherwise does nothing. Raises AssertionError if the IR node is not a ComputedBuffer.
        if not MixOrderReduction.is_split_reduction(self):
            return
        if not isinstance(self.node, ir.ComputedBuffer):
            raise AssertionError("expected self.node to be an ir.ComputedBuffer")
        with self.node.with_original_inner_fn():
            self._compute_attrs()

    def expand_dimension_for_pointwise_node(
        self, dimension: int, new_range: int
    ) -> None:
        # [FUNCTION DOC] Expands the given pointwise dimension to new_range in the body, updates sizes and group, and refreshes
        # dependencies (normalized). Raises AssertionError for unsupported node types.
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(
                "expected self.node to be a ComputedBuffer or TemplateBuffer"
            )

        self._before_loop_state_mutation()
        self._body = self._body.expand_dimension_for_pointwise_node(
            dimension, new_range
        )
        self._sizes = self._body.sizes

        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # Need normalize the prefix name to facilitate finding common dependencies
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=True)

    def merge_loops(self) -> None:
        # [FUNCTION DOC] Merges adjacent loops of the body, updating sizes and refreshing dependencies (normalized, tiling cache kept).
        self._body = self._body.merge_loops()
        self._sizes = self._body.sizes

        # merge_loops is called after loop reordering.
        # We still need retain fake dependencies since codegen the
        # estimated amount of memory access rely on them.
        #
        # Merge loops does not affect the tiling decision. So we
        # don't need clear the tiling cache.
        self.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> bool:
        # [FUNCTION DOC] Reorders loops so this node's access pattern matches other_dep, when both deps have as many variables as the
        # node's iteration sizes and a matching order is found. Returns True (and bumps the loop reordering metric) if
        # reordered, False otherwise.
        new_order = None
        self_sizes = self._sizes[0]
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if new_order:
            # pyrefly: ignore [bad-assignment]
            metrics.num_loop_reordering += 1
            loop_ordering_log.debug(
                "Reorder loops for %s with order %s", self.get_name(), new_order
            )
            self.apply_new_loop_order(new_order)
            return True
        else:
            loop_ordering_log.debug(
                "Don't reordering %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False

    def debug_str_extra(self) -> str:
        # [FUNCTION DOC] Returns extra debug text: group device and iteration, sizes, layout of every non-weak buffer touched, the loop
        # body debug string (if a LoopBody) and device specific lines. Raises AssertionError if node is None.
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        for dep in self.read_writes.reads_and_writes():
            if not isinstance(dep, WeakDep):
                buf_name = dep.name
                buf = V.graph.get_buffer(buf_name)
                if not isinstance(buf, ir.TorchBindObject):
                    lines.append(f"{buf_name}_layout = {pformat(buf.layout)}")
        if isinstance(self._body, LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))

        if self.node is None:
            raise AssertionError("expected self.node to be set")
        lines.extend(self._debug_str_for_device())

        return "\n".join(lines)

    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]:
        # [FUNCTION DOC] Returns the (iteration, reduction) sizes of the node.
        return self._sizes

    def is_reduction(self) -> bool:
        # [FUNCTION DOC] True if the IR node has a reduction type and its loop body has not been converted to pointwise (partial
        # accumulate). Raises AssertionError for node types other than ComputedBuffer or TemplateBuffer.
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(f"{type(self.node)=}")

        # self._body containing partial accumulate means the reduction is
        # converted to a pointwise node.  Need this extra check since
        # we change self._body but didn't change self.node (IRNode)
        # when converting a reduction to a pointwise
        return bool(self.node.get_reduction_type()) and (
            self._body is None or not self._body.has_partial_accumulate
        )

    def is_native_matmul(self) -> bool:
        # [FUNCTION DOC] True if the reduction type is dot (native matmul). Raises AssertionError if the node is not a ComputedBuffer.
        if not isinstance(self.node, ir.ComputedBuffer):
            raise AssertionError(f"{type(self.node)=}")
        return self.node.get_reduction_type() == "dot"

    def is_split_scan(self) -> bool:
        # [FUNCTION DOC] True if the IR node is a ComputedBuffer whose data is a SplitScan. Raises AssertionError for other node types.
        if not isinstance(self.node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            raise AssertionError(f"{type(self.node)=}")
        return isinstance(self.node, ir.ComputedBuffer) and isinstance(
            self.node.data, ir.SplitScan
        )

    def is_template(self) -> bool:
        # [FUNCTION DOC] True if the IR node is a TemplateBuffer.
        return isinstance(self.node, ir.TemplateBuffer)

    def get_template_node(self) -> ir.TemplateBuffer | None:
        # [FUNCTION DOC] Returns the IR node if it is a TemplateBuffer, else None.
        return self.node if isinstance(self.node, ir.TemplateBuffer) else None

    def run(self, *index_vars: Sequence[sympy.Expr]) -> None:
        # [FUNCTION DOC] Generates code for this node: decides inplace updates, marks the node run (allocating outputs), then codegens
        # with the given index variables.
        self.decide_inplace_update()
        self.mark_run()
        self.codegen(index_vars)

    def ranges_from_index_vars(
        self, index_vars: Sequence[Sequence[sympy.Expr]]
    ) -> dict[sympy.Expr, sympy.Expr]:
        # [FUNCTION DOC] HELPER: Maps each index variable to its size by zipping the flattened index_vars with the flattened sizes. Raises
        # AssertionError if the counts differ.
        sizes = self._sizes
        if sum(map(len, sizes)) != sum(map(len, index_vars)):
            raise AssertionError("expected sum of sizes to equal sum of index_vars")
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars: Sequence[Sequence[sympy.Expr]]) -> None:
        # [FUNCTION DOC] Generates kernel code for the node's body with the given index variables, under an indexing-simplifying ops
        # handler built from the variable ranges and with this node set as the kernel's current node. Logs fatally and
        # re-raises any exception.
        """
        Generate code for this node using the provided index variables.

        This method sets up the appropriate context for code generation, including
        simplifying indexing expressions based on the variable ranges, and then
        calls the node's body function with the index variables.

        Args:
            index_vars: A sequence of sequences of sympy expressions representing
                        the index variables for each dimension of the computation.
        """
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with (
                V.set_ops_handler(SimplifyIndexing(V.get_ops_handler(), var_ranges)),
                V.kernel.set_current_node(self),
            ):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_or_reduction_read_writes(
        self, pointwise: bool = True
    ) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Returns read/writes computed over only the pointwise axes (pointwise True) or only the reduction axes, with the
        # other axes' variables fixed to zero.
        """
        Get the memory dependencies in either the pointwise or the reduction axes.
        """
        keep_sizes, ignore_sizes = self._sizes if pointwise else reversed(self._sizes)
        return dependencies.extract_read_writes(
            self._body, keep_sizes, hidden_args=[[sympy.S.Zero] * len(ignore_sizes)]
        )

    @cache_on_self
    def pointwise_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Cached. Memory dependencies over the non-reduction axes.
        """
        Get the memory dependencies in the non-reduction axes.
        """
        return self.pointwise_or_reduction_read_writes(pointwise=True)

    @cache_on_self
    def reduction_read_writes(self) -> dependencies.ReadWrites:
        # [FUNCTION DOC] Cached. Memory dependencies over the reduction axes.
        """
        Get the memory dependencies in the reduction axes.
        """
        return self.pointwise_or_reduction_read_writes(pointwise=False)

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        # [FUNCTION DOC] True if read_dep can be updated in place by this node: not a template, no aliased outputs, exactly one write, and
        # both are MemoryDeps with equal index and size. Raises AssertionError if the write is not a MemoryDep.
        if self.is_template():
            return False
        if any(out.get_aliases() for out in self.get_outputs()):
            return False
        if len(self.read_writes.writes) == 1 and isinstance(
            read_dep, dependencies.MemoryDep
        ):
            write_dep = next(iter(self.read_writes.writes))
            if not isinstance(write_dep, dependencies.MemoryDep):
                raise AssertionError(f"{type(write_dep)=}")
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    @cache_on_self
    def _get_atomic_add_buffers(self) -> OrderedSet[str]:
        # [FUNCTION DOC] HELPER: Cached. Returns the names of buffers stored via atomic add in the loop body (detected from store calls
        # with an atomic_add mode).
        buffers_store_as_atomic_add: OrderedSet[str] = OrderedSet()
        if isinstance(self._body, LoopBody):
            for node in self._body.get_nodes():
                if (
                    node.op == "call_method"
                    and node.target == "store"
                    and (
                        ("mode" in node.kwargs and node.kwargs["mode"] == "atomic_add")
                        or (len(node.args) == 5 and node.args[4] == "atomic_add")
                    )
                ):
                    buffers_store_as_atomic_add.add(
                        node.kwargs["name"]
                        if "name" in node.kwargs
                        else (node.args[1] if len(node.args) >= 2 else "")
                    )
        return buffers_store_as_atomic_add

    @cache_on_self
    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Cached. True if the body contains an async device assert, else the base behavior (False).
        # self._body is None sometimes that's why this check was added
        if self._body is not None and self._body.has_op("device_assert_async"):
            return True
        return super().has_side_effects()


def refresh_group_node_dependencies(
    group_snode: FusedSchedulerNode | GroupedSchedulerNode,
) -> None:
    # [FUNCTION DOC] Recomputes a group node's read/writes as the merge of its sub-nodes, and its unmet dependencies as the union of the
    # sub-nodes' unmet dependencies excluding those satisfied by buffers of the group or its writes.
    snodes = group_snode.snodes
    group_snode.set_read_writes(
        dependencies.ReadWrites.merge_list([x.read_writes for x in snodes])
    )

    group_snode.unmet_dependencies = (
        OrderedSet(
            dep
            for dep in OrderedSet.union(*[x.unmet_dependencies for x in snodes])
            if dep.name not in group_snode.get_buffer_names()
        )
        - group_snode.read_writes.writes
    )


def init_group_node(
    group_snode: FusedSchedulerNode | GroupedSchedulerNode,
    scheduler: Scheduler,
    snodes: list[BaseSchedulerNode],
) -> None:
    # [FUNCTION DOC] Initializes a fused or grouped node from snodes: stores sub-nodes and scheduler, no IR node, union of ancestors,
    # refreshed dependencies, min/max order and input distances over sub-nodes, and outputs by name. Raises AssertionError for
    # other node types.
    if not isinstance(group_snode, (FusedSchedulerNode, GroupedSchedulerNode)):
        raise AssertionError("expected FusedSchedulerNode or GroupedSchedulerNode")
    group_snode.snodes = snodes
    group_snode.scheduler = scheduler
    group_snode.node = None
    group_snode.ancestors = OrderedSet.union(
        *[x.ancestors for x in snodes if x.ancestors is not None]
    )

    refresh_group_node_dependencies(group_snode)

    group_snode.min_order = min(x.min_order for x in group_snode.snodes)
    group_snode.max_order = max(x.max_order for x in group_snode.snodes)
    group_snode.min_input_distance = min(
        x.min_input_distance for x in group_snode.snodes
    )
    group_snode.max_input_distance = max(
        x.max_input_distance for x in group_snode.snodes
    )
    group_snode.outputs_by_name = {
        buf.get_name(): buf for buf in group_snode.get_outputs()
    }


class FusedSchedulerNode(BaseSchedulerNode):
    # [CLASS DOC] Fake scheduler node standing for a list of scheduler nodes fused into one kernel. Holds snodes and users; its group
    # comes from the sub-node preferring reductions. Dependencies are the union of the sub-nodes. Most query methods aggregate
    # over snodes. Mutation-related edits (rename, fake dep, in-place) are not supported and raise NotImplementedError.
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    snodes: list[BaseSchedulerNode]

    @classmethod
    def fuse(
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        # [FUNCTION DOC] Class method. Fuses node1 and node2 (same scheduler; node1 a scheduler or fused node; node2 likewise, or a
        # MultiOutput extern node after a template) into a new fused node over the concatenation of their sub-nodes. Raises
        # AssertionError on violations.
        if node1.scheduler is not node2.scheduler:
            raise AssertionError("expected node1 and node2 to share the same scheduler")
        if not isinstance(node1, (SchedulerNode, FusedSchedulerNode)):
            raise AssertionError(
                "expected node1 to be a SchedulerNode or FusedSchedulerNode"
            )
        if node1.is_template() and isinstance(node2, ExternKernelSchedulerNode):
            if not isinstance(node2.node, ir.MultiOutput):
                raise AssertionError("expected node2.node to be an ir.MultiOutput")
        else:
            if not isinstance(node2, (SchedulerNode, FusedSchedulerNode)):
                raise AssertionError(
                    "expected node2 to be a SchedulerNode or FusedSchedulerNode"
                )
        nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))
        return cls(node1.scheduler, nodes)

    def extract_pw_from_reduction(self) -> BaseSchedulerNode:
        # [FUNCTION DOC] Converts every (reduction) sub-node to its pointwise extraction and returns self. Raises AssertionError if a
        # sub-node is not a reduction SchedulerNode.
        for subnode in self.snodes:
            if not isinstance(subnode, SchedulerNode):
                raise AssertionError("expected subnode to be a SchedulerNode")
            if not subnode.is_reduction():
                raise AssertionError("expected subnode to be a reduction")
            subnode.extract_pw_from_reduction()
        return self

    def swap_pw_red_dimension(self) -> None:
        # [FUNCTION DOC] Swaps pointwise and reduction dimensions of every sub-node (each must be a SchedulerNode, else AssertionError).
        for subnode in self.snodes:
            if not isinstance(subnode, SchedulerNode):
                raise AssertionError("expected subnode to be a SchedulerNode")
            subnode.swap_pw_red_dimension()

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # [FUNCTION DOC] Cached. Sum of FLOP estimates of template and extern sub-nodes (None if none); does not touch the counters to
        # avoid double counting.
        # don't increment counters in fused methods so we don't double count
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def reorder_loops_by_dep_pair(
        self, self_dep: MemoryDep, other_dep: MemoryDep
    ) -> bool:
        # [FUNCTION DOC] Reorders the loops of all sub-nodes to match other_dep when every sub-node is a SchedulerNode with the same sizes
        # and a loop order is found; not for templates. Refreshes group dependencies and bumps a metric. Returns True if reordered.
        """
        Return true if a loop reordering is performed.
        """
        if self.is_template():
            # We can not really reorder loops for a triton template
            return False
        self_sizes = None
        for snode in self.snodes:
            if not isinstance(snode, SchedulerNode):
                return False
            if self_sizes is not None and tuple(self_sizes) != tuple(snode._sizes[0]):
                loop_ordering_log.debug(
                    "Can not reorder fused node due to different sizes"
                )
                return False
            self_sizes = snode._sizes[0]
        new_order = None

        if self_sizes is None:
            raise AssertionError("expected self_sizes to be set")
        if len(self_sizes) == self_dep.num_vars == other_dep.num_vars:
            new_order = self_dep.decide_loop_order_to_match(other_dep)

        if not new_order:
            loop_ordering_log.debug(
                "Don't reordering fused node %s because we can not decide the suitable loop order",
                self.get_name(),
            )
            return False
        # pyrefly: ignore [bad-assignment]
        metrics.num_loop_reordering += 1
        loop_ordering_log.debug(
            "Reorder loops for fused node %s with order %s", self.get_name(), new_order
        )
        for snode in self.snodes:
            if not isinstance(snode, SchedulerNode):
                raise AssertionError("expected snode to be a SchedulerNode")
            snode.apply_new_loop_order(new_order)

        refresh_group_node_dependencies(self)
        return True

    def __init__(self, scheduler: Scheduler, snodes: list[BaseSchedulerNode]) -> None:
        # [FUNCTION DOC] Initializes the base node, group state from snodes, an empty users list, and the group of the first sub-node
        # with highest reduction flag.
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        self.users: list[NodeUser] = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group

    @cache_on_self
    def get_name(self) -> str:
        # [FUNCTION DOC] Cached. Name formed by joining sub-node names with underscores.
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        # [FUNCTION DOC] Returns the name of the first sub-node.
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Cached. Union of sub-node output buffer names.
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list[SchedulerBuffer]:
        # [FUNCTION DOC] Returns the concatenated outputs of all sub-nodes.
        result: list[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    def debug_str_extra(self) -> str:
        # [FUNCTION DOC] Returns indented debug text with each sub-node's debug string and device specific lines.
        lines = [
            f"{self.get_name()}.snodes[{i}] =\n{node.debug_str()}"
            for i, node in enumerate(self.snodes)
        ]
        node = self.snodes[0].node
        if node is not None:
            lines.extend(self._debug_str_for_device())

        return textwrap.indent("\n".join(lines).rstrip(), "    ")

    def debug_str_short(self) -> str:
        # [FUNCTION DOC] Short debug string listing the short debug strings of the sub-nodes.
        snodes_str = [node.debug_str_short() for node in self.snodes]
        return f"{self}, snodes: {snodes_str}"

    def set_last_usage(
        self, future_used_buffers: OrderedSet[str], mutation_real_name: dict[str, str]
    ) -> None:
        # [FUNCTION DOC] Sets the fused node's last_usage using global information, then recomputes last_usage of each sub-node in
        # reverse order using only the buffers used by later sub-nodes (intra-kernel liveness).
        # Set self.last_usage using the global information
        # This will be used for inter-kernel optimisations
        super().set_last_usage(future_used_buffers, mutation_real_name)
        # Set self.last_usage on the snodes
        # This will be used for optimisations within the kernel
        future_used_buffers: OrderedSet[str] = OrderedSet()
        for node in reversed(self.snodes):
            node.set_last_usage(future_used_buffers, mutation_real_name)
            future_used_buffers.update(node.last_usage)

    @cache_on_self
    def used_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Cached. Union of buffer names used by sub-nodes.
        return OrderedSet.union(*[x.used_buffer_names() for x in self.snodes])

    @cache_on_self
    def used_or_aliased_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Cached. Union of used-or-aliased buffer names of sub-nodes.
        return OrderedSet.union(
            *[x.used_or_aliased_buffer_names() for x in self.snodes]
        )

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        # [FUNCTION DOC] Returns the sub-nodes.
        return self.snodes

    def __repr__(self) -> str:
        # [FUNCTION DOC] Repr with class name and the fused node name.
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node is a reduction.
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_native_matmul(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node is a native matmul.
        return any(x.is_native_matmul() for x in self.snodes)

    @cache_on_self
    def is_split_scan(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node is a split scan.
        return any(x.is_split_scan() for x in self.snodes)

    @cache_on_self
    def is_template(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node is a template.
        return any(x.is_template() for x in self.snodes)

    @cache_on_self
    def get_template_node(self) -> ir.TemplateBuffer | None:
        # [FUNCTION DOC] Cached. Template buffer of the first template sub-node, else None.
        for node in self.snodes:
            if node.is_template():
                return node.get_template_node()
        return None

    def get_device(self) -> torch.device:
        # [FUNCTION DOC] Returns the device from the group.
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node aliases or mutates.
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: dict[str, str]) -> None:
        # [FUNCTION DOC] Not supported for fused nodes: raises NotImplementedError.
        raise NotImplementedError

    def add_fake_dep(self, name: Dep) -> None:
        # [FUNCTION DOC] Not supported for fused nodes: raises NotImplementedError.
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.Dep) -> bool:
        # [FUNCTION DOC] Not supported for fused nodes: raises NotImplementedError.
        raise NotImplementedError

    def debug_str(self) -> str:
        # [FUNCTION DOC] Longer debug printout listing sub-node types, writes, dependencies, distances and outputs, plus extra debug text
        # (errors in the extra text are logged and ignored).
        """Longer form printout for trace logs"""
        name = self.get_name()
        node_typestr = ",".join(type(n).__name__ for n in self.snodes)
        buf = IndentedBuffer()
        buf.splice(
            f"""\
{name}: {type(self).__name__}({node_typestr})
{name}.writes = {pformat(self.read_writes.writes)}
{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}
{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}
{name}.min_input_distance = {self.min_input_distance}
{name}.max_input_distance = {self.max_input_distance}
{name}.outputs = [
            """
        )
        with buf.indent():
            for out in self.get_outputs():
                buf.splice(out.debug_str())
        buf.writeline("]")

        try:
            buf.splice(self.debug_str_extra())
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)

        return buf.getrawvalue().rstrip()

    @cache_on_self
    def has_side_effects(self) -> bool:
        # [FUNCTION DOC] Cached. True if any sub-node has side effects.
        if self.snodes is not None:
            return any(node.has_side_effects() for node in self.snodes)
        return super().has_side_effects()


class FusedMixOrderReductions(FusedSchedulerNode):
    # [CLASS DOC] Fused node for two reductions with different iteration orders (the contiguous one is node1, the other node2),
    # as produced by mix order reduction fusion. Keeps node1, node2 and numel of node1. Supports fusing further nodes into
    # either side without creating producer/consumer links between the two sides.
    """Fused node for two reductions with different iteration orders (inner + outer)."""

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] Orders the two reductions so node1 is contiguous (raises AssertionError if neither is), stores them, initializes
        # the fused node over all their sub-nodes and records numel.
        if not MixOrderReduction.is_contiguous_node(node1):
            if not MixOrderReduction.is_contiguous_node(node2):
                raise AssertionError("expected node2 to be a contiguous node")
            node1, node2 = node2, node1

        self.node1 = node1
        self.node2 = node2
        super().__init__(
            node1.scheduler, list(node1.get_nodes()) + list(node2.get_nodes())
        )
        self.numel = MixOrderReduction.get_numel(self.node1)

    def sub_node_can_fuse(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        other_nodes: tuple[BaseSchedulerNode, ...],
    ):
        # [FUNCTION DOC] Checks whether node2 can fuse into node1 (a side of this mix order reduction): scheduler legality without
        # recursive mix-order fusion, contiguity preserved, no producer/consumer relation between the two sides (via other_nodes),
        # and for reductions a fusion memory score of at least numel. Raises AssertionError if either node is already a mix order node.
        """
        node1 is from the current mix order reduction; node2 is another node we want to fuse in.

        other_nodes are passed in to check if fusion will introduce producer/consumer relationship
        between the inner and outer reduction. If yes, we don't fuse.
        """
        if isinstance(node1, FusedMixOrderReductions):
            raise AssertionError("expected node1 to not be a FusedMixOrderReductions")
        if isinstance(node2, FusedMixOrderReductions):
            raise AssertionError("expected node2 to not be a FusedMixOrderReductions")

        # When we fuse extra nodes into a FusedMixOrderReductions node,
        # we should not allow recursive mix-order reduction being
        # created.
        if not self.scheduler.can_fuse(node1, node2, allow_mix_order_reduction=False):
            return False

        # Since node1 is from the current mix order reduction, if node1 is
        # contiguous, the fused node should also be contiguous.
        if MixOrderReduction.is_contiguous_node(
            node1
        ) and not MixOrderReduction.is_contiguous_node(node2):
            return False

        def _get_ancestors(nodes: tuple[BaseSchedulerNode, ...]) -> OrderedSet[str]:
            # [FUNCTION DOC] HELPER: Union of the ancestors of the given nodes.
            out = OrderedSet()
            return out.union(*(n.ancestors for n in nodes))

        def _get_operation_names(
            nodes: tuple[BaseSchedulerNode, ...],
        ) -> OrderedSet[str]:
            # [FUNCTION DOC] HELPER: Union of the operation names of the given nodes.
            out = OrderedSet()
            return out.union(*(n.get_operation_names() for n in nodes))

        if other_nodes:
            if (_get_ancestors((node1, node2)) & _get_operation_names(other_nodes)) or (
                _get_ancestors(other_nodes) & _get_operation_names((node1, node2))
            ):
                return False

        return (
            not node2.is_reduction()
            or self.scheduler.score_fusion_memory(node1, node2, count_bytes=False)
            >= self.numel
        )

    def can_fuse_with(self, other: BaseSchedulerNode):
        # [FUNCTION DOC] Whether other can be fused into this mix order reduction. False with strict sum, or when the combined set of
        # read buffers exceeds the configured max reads (bumps a rejection metric). A plain node must fuse into either side; another
        # mix order node needs both sides to be fusable pairwise.
        if self.has_strict_sum() or other.has_strict_sum():
            return False
        # Limit tl.load() count in the fused RSPLIT loop to avoid register
        # spills. See https://github.com/pytorch/pytorch/issues/179423
        max_reads = config.triton.mix_order_reduction_max_reads
        if max_reads > 0:
            all_reads: OrderedSet[str] = OrderedSet()
            for sn in itertools.chain(self.get_nodes(), other.get_nodes()):
                for dep in sn.read_writes.reads:
                    if isinstance(dep, MemoryDep):
                        all_reads.add(dep.name)
            if len(all_reads) > max_reads:
                # pyrefly: ignore [bad-assignment]
                metrics.rejected_mix_order_reduction_fusion += 1
                return False
        if not isinstance(other, FusedMixOrderReductions):
            return self.sub_node_can_fuse(
                self.node1, other, (self.node2,)
            ) or self.sub_node_can_fuse(self.node2, other, (self.node1,))
        else:
            # pass empty tuple for the second since the producer/consumer relationship has
            # already been checked in the first call
            return self.sub_node_can_fuse(
                self.node1, other.node1, (self.node2, other.node2)
            ) and self.sub_node_can_fuse(self.node2, other.node2, tuple())

    def fuse_with(self, other: BaseSchedulerNode):
        # [FUNCTION DOC] Fuses other into this node using the backend fuse and returns a new FusedMixOrderReductions: pairwise fusion of
        # sides for another mix order node, otherwise into the first side when allowed, else the second.
        device = self.node1.get_device()
        backend = self.scheduler.get_backend(device)

        if isinstance(other, FusedMixOrderReductions):
            fused_node1 = backend.fuse(self.node1, other.node1)
            fused_node2 = backend.fuse(self.node2, other.node2)
            return FusedMixOrderReductions(fused_node1, fused_node2)
        else:
            if self.sub_node_can_fuse(self.node1, other, (self.node2,)):
                fused_node = backend.fuse(self.node1, other)
                return FusedMixOrderReductions(fused_node, self.node2)
            else:
                fused_node = backend.fuse(self.node2, other)
                return FusedMixOrderReductions(self.node1, fused_node)


class FusedNestedReductions(FusedSchedulerNode):
    # [CLASS DOC] Fused node for two dependent reductions over the same logical elements (outer reduction plus grouped local
    # reduction staged inside the outer grid). Stores node1/node2, the grouped reduction, exact group size, grouped axis,
    # whether the group is in R, and the pointwise domain context.
    """
    Fused node for two dependent reductions over the same logical elements.
    The outer reduction owns the codegen grid; the grouped reduction is staged
    inside that grid.
    """

    def __init__(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] Builds the fused node from outer node1 and grouped node2, removes internal producer/consumer ancestors, and
        # derives grouped reduction, group size, grouped axis and domain context. Raises AssertionError if the grouped reduction or
        # axis cannot be determined.
        self.node1 = node1
        self.node2 = node2
        super().__init__(
            node1.scheduler, list(node1.get_nodes()) + list(node2.get_nodes())
        )
        # Nested append fusion checks legality against node2, the grouped
        # stage, while the fused node owns producers from both stages. Hide
        # internal producer-consumer edges from later ancestor checks so reads
        # from node1 are not mistaken for external intermediate deps.
        self.ancestors -= self.get_operation_names()
        grouped_node = node2
        _, (grouped_numel, grouped_rnumel) = grouped_node.group
        grouped_reduction_info = NestedReduction._get_grouped_reduction_and_size(
            grouped_node, grouped_rnumel
        )
        if grouped_reduction_info is None:
            raise AssertionError("expected grouped_reduction_info to be non-None")
        grouped_reduction, exact_group_size = grouped_reduction_info
        self.grouped_reduction: SchedulerNode = grouped_reduction
        self.group_size: sympy.Integer = exact_group_size
        _, (outer_numel, outer_rnumel) = node1.group
        grouped_axis = NestedReduction.get_grouped_axis(
            grouped_reduction,
            outer_numel,
            outer_rnumel,
            exact_group_size,
            outer_node=node1,
        )
        if grouped_axis is None:
            raise AssertionError("expected grouped_axis to be non-None")
        self.grouped_axis: NestedReduction.GroupedAxis = grouped_axis
        self.group_size_in_r: bool = self.grouped_axis is NestedReduction.GroupedAxis.R
        iter_ranges, reduce_ranges = grouped_reduction.get_ranges()
        self.domain_context: NestedReduction.PointwiseDomainContext = (
            NestedReduction.PointwiseDomainContext(
                grouped_reduction=grouped_reduction,
                grouped_numel=grouped_numel,
                grouped_rnumel=grouped_rnumel,
                local_reduction_domain=(*iter_ranges, *reduce_ranges),
                parent_full_domain=(outer_numel, outer_rnumel),
            )
        )

    def can_fuse_with(self, other: BaseSchedulerNode, *, can_reorder: bool) -> bool:
        # [FUNCTION DOC] Decides whether pointwise node other (downstream of the grouped reduction) can fuse in: it must not be a
        # reduction, must consume the grouped stage and its pointwise sub-nodes must classify into compatible nested domains.
        """Allow downstream pointwise of the grouped reduction to fuse in.

        Consumers fused directly into the grouped reduction must run at either
        reduced-output resolution or full parent-tile resolution.
        """
        if other.is_reduction():
            return False
        # fuse_with() appends into node2, the grouped stage. Pointwise nodes
        # that do not consume that stage need a different insertion point.
        if not (self.node2.get_operation_names() & other.ancestors):
            return False

        pointwise_domains: (
            list[tuple[SchedulerNode, NestedReduction.PointwiseDomain]] | None
        ) = NestedReduction._classify_grouped_pointwise_nodes(
            self.domain_context,
            other.get_nodes(),
        )
        if pointwise_domains is None:
            return False
        if not NestedReduction._pointwise_domains_are_compatible(
            self.domain_context, pointwise_domains
        ):
            return False
        # Producers for the grouped reduction body need to be inserted before
        # the reduction; this append-only path only accepts consumers.
        if any(
            domain is NestedReduction.PointwiseDomain.LOCAL_REDUCTION_INPUT
            for _, domain in pointwise_domains
        ):
            return False
        return self.scheduler._can_fuse_nested_reduction_append(
            self.node2,
            other,
            pointwise_domains,
            can_reorder=can_reorder,
        )

    def fuse_with(self, other: BaseSchedulerNode) -> FusedNestedReductions:
        # [FUNCTION DOC] Fuses other into the grouped stage (node2) via the backend fuse and returns a new FusedNestedReductions with
        # unchanged node1 and the extended node2.
        device = self.node2.get_device()
        backend = self.scheduler.get_backend(device)
        new_node2 = backend.fuse(self.node2, other)
        return FusedNestedReductions(self.node1, new_node2)


class FusedExternTritonKernelSchedulerNode(FusedSchedulerNode):
    # [CLASS DOC] Fused node combining a user-defined Triton extern kernel node with a pointwise epilogue SchedulerNode that
    # consumes its mutated output. Behaves as an extern node; outputs are the epilogue's outputs; order comes from the kernel node.
    def __init__(
        self,
        scheduler: Scheduler,
        kernel_node: ExternKernelSchedulerNode,
        fused_epilogue: SchedulerNode,
    ) -> None:
        # [FUNCTION DOC] Builds the fused node over [kernel_node, fused_epilogue] and stores both; raises AssertionError if the kernel
        # node is not a UserDefinedTritonKernel.
        if not isinstance(kernel_node.node, ir.UserDefinedTritonKernel):
            raise AssertionError(
                "expected kernel_node.node to be an ir.UserDefinedTritonKernel"
            )
        snodes = typing.cast(list[BaseSchedulerNode], [kernel_node, fused_epilogue])
        super().__init__(scheduler, snodes)
        self.kernel_node = kernel_node
        self.fused_epilogue = fused_epilogue
        self.min_order = self.kernel_node.min_order
        self.outputs = fused_epilogue.outputs

    @classmethod
    def epilogue_fuse(
        cls,
        node1: ExternKernelSchedulerNode,
        node2: SchedulerNode,
    ) -> FusedSchedulerNode:
        # [FUNCTION DOC] Class method. Creates the fused node from a Triton kernel node and its epilogue. Side effect: removes node1 from the
        # users of its (real-named) mutated buffer, because that buffer becomes an intermediate and dead node elimination should be able
        # to drop it. Raises AssertionError unless the kernel is user-defined with exactly one mutation output.
        if not isinstance(node1.node, ir.UserDefinedTritonKernel):
            raise AssertionError(
                "expected node1.node to be an ir.UserDefinedTritonKernel"
            )
        scheduler = node1.scheduler

        if len(node1.node.mutation_outputs) != 1:
            raise AssertionError(
                f"expected one mutation output, got {len(node1.node.mutation_outputs)}"
            )
        # pyrefly: ignore[bad-assignment]
        mutated_name: str = node1.node.mutation_outputs[0].name
        # Node1's mutated tensor becomes an intermediary tensor.
        # Thus, remove node1 from the respective allocated buffer's users
        # for `Scheduler.dead_node_elimination` to remove.
        real_name = scheduler.mutation_real_name.get(mutated_name, mutated_name)
        scheduler.name_to_buf[real_name].users.remove(NodeUser(node1))
        return cls(scheduler, node1, node2)

    def codegen(self, wrapper: PythonWrapperCodegen) -> None:
        # [FUNCTION DOC] Generates the kernel source of the epilogue fused into the user kernel (SIMD tiling and features for the epilogue,
        # FusedUserDefinedTritonKernel) and emits the user kernel with epilogue fusion through the wrapper. Raises AssertionError if
        # node types are wrong or the kernel does not allow epilogue fusion. (Annotated as returning None but returns the codegen result.)
        if not isinstance(self.fused_epilogue.node, ir.ComputedBuffer):
            raise AssertionError(
                "expected fused_epilogue.node to be an ir.ComputedBuffer"
            )
        if not isinstance(self.kernel_node.node, ir.UserDefinedTritonKernel):
            raise AssertionError(
                "expected kernel_node.node to be an ir.UserDefinedTritonKernel"
            )
        if not self.kernel_node.node.can_fuse_epilogue():
            raise AssertionError("expected kernel_node.node to allow epilogue fusion")
        numel = math.prod(self.kernel_node.node.mutable_args[0].shape)
        from torch._inductor.codegen.simd import SIMDScheduling

        tiling, _ = SIMDScheduling.get_tiling_and_scores([self.fused_epilogue], numel)

        from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures

        kernel_features = SIMDKernelFeatures([self.fused_epilogue], numel)

        from torch._inductor.codegen.triton import FusedUserDefinedTritonKernel

        fused_user_kernel = FusedUserDefinedTritonKernel(tiling, kernel_features, self)
        new_kernel_src = fused_user_kernel.codegen()

        return self.kernel_node.node.codegen_with_epilogue_fusion(
            wrapper, (self.fused_epilogue.node, new_kernel_src)
        )

    def is_extern(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def get_ranges(self) -> Sequence[Sequence[sympy.Expr]]:
        # [FUNCTION DOC] Returns the ranges of the kernel node.
        return self.kernel_node.get_ranges()


class ForeachKernelSchedulerNode(FusedSchedulerNode):
    # [CLASS DOC] Combo-kernel node: a set of scheduler nodes (possibly vertically fused) with no data dependencies among the
    # top-level members, executed in one kernel. Keeps maps from read buffer name to member and from operation name to member,
    # the custom partition and autotune flags, per-subkernel block flag and a synthetic combo group. Provides combo candidate
    # filtering, grouping algorithm (replaceable), and fusion of regular nodes into one member.
    """
    This is a schedular node that consists of a set of scheduler nodes that
    has no data dependencies among them and can be executed in parallel.
    """

    def get_consumer_subnode_for(
        self, producer: BaseSchedulerNode
    ) -> BaseSchedulerNode | None:
        # [FUNCTION DOC] Returns the member node that reads any output of producer, or None.
        for buf in producer.get_outputs():
            if buf.get_name() in self.read_to_node:
                return self.read_to_node[buf.get_name()]

        return None

    def get_producer_subnode_for(
        self, consumer: BaseSchedulerNode
    ) -> BaseSchedulerNode | None:
        # [FUNCTION DOC] Returns the unique member node defining a buffer read by consumer; None if there are zero or multiple such members.
        producers = OrderedSet[BaseSchedulerNode]()
        for rd in consumer.read_writes.reads:
            if rd.name not in self.scheduler.name_to_buf:
                continue

            node_name = self.scheduler.name_to_buf[rd.name].defining_op_name()
            if node_name in self.name_to_node:
                producers.add(self.name_to_node[node_name])

        # Don't permit fusion if there are multiple subnodes
        # that this consumer reads from
        if len(producers) == 1:
            return next(iter(producers))
        else:
            return None

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] Class method. Decides if producer and consumer can be fused where at least one is foreach (else raises
        # AssertionError). Two foreach nodes: same length and pairwise fusable. Otherwise the non-foreach node must not be a reduction
        # and must depend on (or be consumed by) a member, and the scheduler must allow fusing with that member. Logs reasons with WhyNoFuse.
        why = WhyNoFuse(producer, consumer)
        if producer.is_foreach() and consumer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            foreach_match = len(producer.snodes) == len(consumer.snodes)
            if not foreach_match:
                why("foreach do not have same length")
            return foreach_match and all(
                producer.scheduler.can_fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            )
        elif consumer.is_foreach():
            if producer.is_reduction():
                why(
                    "candidate producer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            if consumer_subnode is not None:
                return consumer.scheduler.can_fuse(producer, consumer_subnode)

            why("candidate producer is not dep of any foreach consumer")
            return False

        elif producer.is_foreach():
            if consumer.is_reduction():
                why(
                    "candidate consumer is a reduction, foreach ops cannot be fused with reductions currently"
                )
                return False

            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            if producer_subnode is not None:
                return producer.scheduler.can_fuse(producer_subnode, consumer)

            why("candidate consumer has no dep in any foreach producer")
            return False

        raise AssertionError(
            "At least one node passed to ForeachKernelSchedulerNode.can_fuse should be a foreach node"
        )

    @classmethod
    def fuse(
        cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> ForeachKernelSchedulerNode:
        # [FUNCTION DOC] Class method. Fuses producer and consumer, at least one a foreach node (else AssertionError): pairwise fusion for two
        # foreach nodes, or fuses the other node into its related member while keeping the rest. Returns a new foreach node keeping the
        # custom partition and autotune settings, with previous-node info so dependencies are merged cheaply.
        if not (producer.is_foreach() or consumer.is_foreach()):
            raise AssertionError("expected producer or consumer to be foreach")
        if producer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            use_custom_partition_algo = producer.use_custom_partition_algo
            enable_autotune = producer.enable_autotune
        else:
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            use_custom_partition_algo = consumer.use_custom_partition_algo
            enable_autotune = consumer.enable_autotune
        prev_node_1 = None
        prev_node_2 = None
        fused_nodes: list[BaseSchedulerNode]
        if producer.is_foreach() and consumer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            fused_nodes = [
                FusedSchedulerNode.fuse(l, r)
                for l, r in zip(producer.snodes, consumer.snodes)
            ]
        elif producer.is_foreach():
            producer = typing.cast(ForeachKernelSchedulerNode, producer)
            producer_subnode = producer.get_producer_subnode_for(consumer)
            fused_nodes = []
            prev_node_1 = producer
            prev_node_2 = None
            for node in producer.snodes:
                if node is producer_subnode:
                    new_node = FusedSchedulerNode.fuse(node, consumer)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)

        elif consumer.is_foreach():
            consumer = typing.cast(ForeachKernelSchedulerNode, consumer)
            consumer_subnode = consumer.get_consumer_subnode_for(producer)
            fused_nodes = []
            prev_node_1 = consumer
            prev_node_2 = None

            for node in consumer.snodes:
                if node is consumer_subnode:
                    new_node = FusedSchedulerNode.fuse(producer, node)
                    prev_node_2 = new_node
                    fused_nodes.append(new_node)
                else:
                    fused_nodes.append(node)
        else:
            raise AssertionError(
                "At least one node passed to ForeachKernelSchedulerNode.fuse should be a foreach node"
            )

        return cls(
            producer.scheduler,
            fused_nodes,
            use_custom_partition_algo=use_custom_partition_algo,
            prev_node_1=prev_node_1,
            prev_node_2=prev_node_2,
            enable_autotune=enable_autotune,
        )

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: list[BaseSchedulerNode],
        use_custom_partition_algo: bool,
        prev_node_1: BaseSchedulerNode | None = None,
        prev_node_2: BaseSchedulerNode | None = None,
        enable_autotune: bool = False,
        per_subkernel_blocks: bool = False,
    ) -> None:
        # [FUNCTION DOC] Builds the combo node. Without prior foreach info, initializes as a fused node and indexes members by read buffer and
        # operation name. With prev_node_1 and prev_node_2, merges their read/writes, unmet dependencies, orders, distances, ancestors and
        # name map (reuse of the existing foreach node's map). Then sets the device-based combo group, origins, autotune and
        # per-subkernel-block flags. Raises AssertionError on a missing device or bad prior nodes.
        self.read_to_node = {}
        self.name_to_node = {}

        if prev_node_1 is None or prev_node_2 is None:
            super().__init__(scheduler, snodes)

            for node in snodes:
                for read in node.read_writes.reads:
                    self.read_to_node[read.name] = node

                for name in node.get_operation_names():
                    self.name_to_node[name] = node
        else:
            self.scheduler = scheduler
            self.snodes = snodes
            self.node = None
            self.users: list[NodeUser] = []

            self.set_read_writes(
                dependencies.ReadWrites.merge_list(
                    [prev_node_1.read_writes, prev_node_2.read_writes]
                )
            )

            self.unmet_dependencies = (
                OrderedSet(
                    dep
                    for dep in OrderedSet.union(
                        prev_node_1.unmet_dependencies, prev_node_2.unmet_dependencies
                    )
                    if dep.name not in self.get_buffer_names()
                )
                - self.read_writes.writes
            )

            self.min_order = min([prev_node_1.min_order, prev_node_2.min_order])
            self.max_order = max([prev_node_1.max_order, prev_node_2.max_order])
            self.min_input_distance = min(
                prev_node_1.min_input_distance, prev_node_2.min_input_distance
            )
            self.max_input_distance = max(
                prev_node_1.max_input_distance, prev_node_2.max_input_distance
            )

            if prev_node_1.is_foreach():
                if not isinstance(prev_node_1, ForeachKernelSchedulerNode):
                    raise AssertionError(
                        "expected prev_node_1 to be a ForeachKernelSchedulerNode"
                    )
                foreach_node, other_node = prev_node_1, prev_node_2
            else:
                if not isinstance(prev_node_2, ForeachKernelSchedulerNode):
                    raise AssertionError(
                        "expected prev_node_2 to be a ForeachKernelSchedulerNode"
                    )
                foreach_node, other_node = prev_node_2, prev_node_1

            self.ancestors = foreach_node.ancestors
            self.ancestors.update(other_node.ancestors)

            self.name_to_node = foreach_node.name_to_node
            for name in other_node.get_operation_names():
                self.name_to_node[name] = other_node

            self.outputs_by_name: dict[str, SchedulerBuffer] = {
                k: v for snode in self.snodes for k, v in snode.outputs_by_name.items()
            }

        self.use_custom_partition_algo = use_custom_partition_algo
        device = snodes[0].get_device()
        if not device:
            raise AssertionError("expected device to be set")
        self.group = (device, ((sympy.Expr("combo_kernel"),),))
        self.origins = OrderedSet[torch.fx.Node]()
        self.enable_autotune = enable_autotune
        self.per_subkernel_blocks = per_subkernel_blocks

    @classmethod
    def combinable_nodes(
        cls, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Class method. Filters nodes down to combo-kernel candidates, dropping (with debug logs) no-op, extern, grouped, mix order,
        # nested reduction, existing foreach, template and strict-sum nodes; reductions if combo_kernels_pointwise_only; and nodes with
        # indirect indexing when benchmarking or compile-time autotuning combos. Returns the remaining nodes in order.
        """Filter a node list down to combo-kernel candidates.

        Drops node kinds that can't or shouldn't share a combo kernel:
        extern, grouped, mixed-order reduction, existing foreach, and
        template nodes.
        """
        extern = [x for x in nodes if isinstance(x, ExternKernelSchedulerNode)]
        if extern:
            log.debug(
                "ComboKernels: %d external nodes are filtered %s",
                len(extern),
                [node.node.get_origins() for node in extern if node.node is not None],
            )
        grouped = [x for x in nodes if isinstance(x, GroupedSchedulerNode)]
        if grouped:
            log.debug(
                "ComboKernels: %d grouped nodes are filtered",
                len(grouped),
            )
        mix_order = [x for x in nodes if isinstance(x, FusedMixOrderReductions)]
        if mix_order:
            log.debug(
                "ComboKernels: %d FusedMixOrderReductions nodes are filtered",
                len(mix_order),
            )
        nested_reductions = [x for x in nodes if isinstance(x, FusedNestedReductions)]
        if nested_reductions:
            log.debug(
                "ComboKernels: %d FusedNestedReductions nodes are filtered",
                len(nested_reductions),
            )

        filtered_nodes = [
            x
            for x in nodes
            if not isinstance(
                x,
                (
                    NopKernelSchedulerNode,
                    ExternKernelSchedulerNode,
                    GroupedSchedulerNode,
                    FusedMixOrderReductions,
                    FusedNestedReductions,
                ),
            )
        ]
        foreach_nodes = [
            x for x in filtered_nodes if isinstance(x, ForeachKernelSchedulerNode)
        ]
        if foreach_nodes:
            log.debug("ComboKernels: %d foreach nodes are filtered", len(foreach_nodes))
        filtered_nodes = [
            x for x in filtered_nodes if not isinstance(x, ForeachKernelSchedulerNode)
        ]
        template_nodes = [x for x in filtered_nodes if x.is_template()]
        if template_nodes:
            log.debug(
                "ComboKernels: %d template nodes are filtered: %s",
                len(template_nodes),
                template_nodes,
            )
        filtered_nodes = [x for x in filtered_nodes if x not in template_nodes]

        # Keep strict sums standalone so their planned R0_BLOCK cannot change.
        filtered_nodes = [node for node in filtered_nodes if not node.has_strict_sum()]

        # Filter out reduction nodes if combo_kernels_pointwise_only is enabled
        if config.combo_kernels_pointwise_only:
            reduction_nodes = [x for x in filtered_nodes if x.is_reduction()]
            if reduction_nodes:
                log.debug(
                    "ComboKernels: %d reduction nodes are filtered (pointwise_only mode)",
                    len(reduction_nodes),
                )
            filtered_nodes = [x for x in filtered_nodes if not x.is_reduction()]

        # Synthetic benchmark inputs cannot preserve indirect-index constraints and may
        # produce out-of-bounds indices. Keep these nodes out of benchmarked combos.
        if config.benchmark_combo_kernel or (
            config.combo_kernels_autotune > 0
            and config.combo_kernel_per_subkernel_blocks
            and config.combo_kernel_compile_time_autotune
        ):
            indirect_nodes = [
                n
                for n in filtered_nodes
                if any(
                    isinstance(dep, MemoryDep) and dep.is_indirect()
                    for dep in n.read_writes.reads_and_writes()
                )
            ]
            if indirect_nodes:
                log.debug(
                    "ComboKernels: %d indirect-indexing nodes are filtered",
                    len(indirect_nodes),
                )
                filtered_nodes = [n for n in filtered_nodes if n not in indirect_nodes]

        return filtered_nodes

    @staticmethod
    def _default_group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> list[list[BaseSchedulerNode]]:
        # [FUNCTION DOC] Static. Default grouping of nodes into combo kernel groups: topologically sorted node levels are split by device (skipping
        # mps and cpu and nodes reading mix order outputs), then by stream and mempool, then chunked into lists of at most
        # config.combo_kernel_max_num_nodes. Returns a list of node lists.
        """
        Returns a list of lists of nodes that are to be grouped together.
        """
        sorted_nodes = scheduler._topological_sort_nodes()
        grouped_nodes = []
        max_num_nodes = config.combo_kernel_max_num_nodes

        excluded_buffer_names: OrderedSet[str] = OrderedSet(
            [
                buf_name
                for group in sorted_nodes
                for node in group
                if isinstance(node, FusedMixOrderReductions)
                for buf_name in node.get_buffer_names()
            ]
        )
        for nodes in sorted_nodes:
            # Group nodes by device first to avoid mixed-device fusion
            device_groups: dict[torch.device | None, list[BaseSchedulerNode]] = (
                defaultdict(list)
            )
            for node in nodes:
                device = node.get_device()
                if device and (device.type == "mps" or device.type == "cpu"):
                    continue

                # exclude nodes that read from FusedMixOrderReductions output buffers'
                if node.used_buffer_names() & excluded_buffer_names:
                    continue
                device_groups[device].append(node)

            # Sub-group by stream and mempool to avoid mixing nodes across
            # context boundaries. When a context is inactive every node maps to
            # the default key so this is a no-op.
            for device_nodes in device_groups.values():
                context_groups: dict[
                    tuple[int, tuple[int, int] | None], list[BaseSchedulerNode]
                ] = defaultdict(list)
                for node in device_nodes:
                    context_groups[
                        (
                            scheduler.get_node_stream(node),
                            scheduler.node_to_mempool.get(node),
                        )
                    ].append(node)
                for context_nodes in context_groups.values():
                    grouped_nodes.extend(
                        [
                            context_nodes[i : i + max_num_nodes]
                            for i in range(0, len(context_nodes), max_num_nodes)
                        ]
                    )
        return grouped_nodes

    group_algorithm_for_combo_kernels: Callable[
        [Scheduler], list[list[BaseSchedulerNode]]
    ] = _default_group_nodes_for_combo_kernels

    @staticmethod
    def set_group_algorithm_for_combo_kernels(
        custom_group_algorithm: Callable[[Scheduler], list[list[BaseSchedulerNode]]],
    ) -> None:
        # [FUNCTION DOC] Static. Replaces the class-wide grouping algorithm used for combo kernels with custom_group_algorithm (global side effect).
        ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels = (
            custom_group_algorithm
        )

    @staticmethod
    def group_nodes_for_combo_kernels(
        scheduler: Scheduler,
    ) -> list[list[BaseSchedulerNode]]:
        # [FUNCTION DOC] Static. Applies the currently configured grouping algorithm to the scheduler and returns lists of nodes to group.
        return ForeachKernelSchedulerNode.group_algorithm_for_combo_kernels(scheduler)

    def mark_run(self) -> None:
        # [FUNCTION DOC] Not supported: raises NotImplementedError.
        raise NotImplementedError

    def codegen(self) -> None:
        # [FUNCTION DOC] Not supported on the node itself: raises NotImplementedError.
        raise NotImplementedError

    def is_foreach(self) -> bool:
        # [FUNCTION DOC] Always True.
        return True

    def get_subkernel_nodes(self) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Returns a copy of the list of member nodes (which may be vertically fused).
        """Returns a list of nodes which comprise the combo kernel.
        These nodes may be vertically fused."""
        return list(self.snodes)

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        # [FUNCTION DOC] Returns all scheduler nodes in the kernel, flattening fused members.
        """Returns all nodes contained in this kernel, unpacking fused nodes
        into their constituent scheduler nodes."""
        return list(itertools.chain.from_iterable(x.get_nodes() for x in self.snodes))

    def get_first_name(self) -> str:
        # [FUNCTION DOC] Returns the first name of the first member.
        return self.snodes[0].get_first_name()

    def prune_redundant_deps(
        self, name_to_fused_node: dict[str, BaseSchedulerNode]
    ) -> None:
        # [FUNCTION DOC] Prunes redundant dependencies of this node and then of every member node.
        _prune_redundant_deps(self, name_to_fused_node, self.scheduler.name_to_buf)

        for node in self.snodes:
            node.prune_redundant_deps(name_to_fused_node)


class GroupedSchedulerNode(BaseSchedulerNode):
    # [CLASS DOC] Fake node grouping scheduler nodes so they stay contiguous in the schedule and no other node may fuse into them
    # or be scheduled between them. Dependencies are the union of members. Fusion still happens among members when unpacked. A temp_grouping
    # flag marks temporary groups used by passes (no fusion on unpack).
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be *grouped* together (it does not allow another node to be scheduled
    in between its constituent nodes, nor does it allow another node to fuse into any of its constituent nodes).
    The way it does this is by maintaining its unmet dependencies as the union of its constituent nodes.
    Fusion will still happen among the nodes within each GroupedSchedulerNode.
    At codegen time, this scheduler node will be unpacked and codegen is called on each constituent node.
    """

    snodes: list[BaseSchedulerNode]

    @classmethod
    def create(cls, snodes: list[BaseSchedulerNode]) -> GroupedSchedulerNode:
        # [FUNCTION DOC] Class method. Creates a group of snodes (all must share a scheduler, else AssertionError) and registers the group and
        # each member in the scheduler's name to fused node map (side effect). Returns the group.
        scheduler = snodes[0].scheduler
        if not all(node.scheduler is scheduler for node in snodes):
            raise AssertionError("expected all nodes to share the same scheduler")
        grouped_snode = cls(scheduler, snodes)
        for snode in snodes:
            scheduler.name_to_fused_node[snode.get_name()] = grouped_snode
        scheduler.name_to_fused_node[grouped_snode.get_name()] = grouped_snode
        return grouped_snode

    def __init__(
        self,
        scheduler: Scheduler,
        snodes: list[BaseSchedulerNode],
        temp_grouping: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initializes the group from snodes and records temp_grouping.
        super().__init__(scheduler)
        init_group_node(self, scheduler, snodes)
        # This flag is introduced for "temporary" grouping during some passes,
        # Where nodes are grouped and moved together.
        # After the pass those nodes are flattened.
        # Reusing calculation of grouped unmed_dependencies etc.
        # No fusion logic in this case.
        self.temp_grouping = temp_grouping

    def unpack(self) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Unpacks the group. A temporary group simply returns its members. Otherwise re-maps each member name to itself in the
        # scheduler's fused node map, removes the group's entry (side effects), and returns the result of running fusion among the members.
        """
        Do fusion among nodes within this GroupedSchedulerNode,
        and then unpack this GroupedSchedulerNode into regular nodes.
        """
        if self.temp_grouping:
            return self.snodes

        for snode in self.snodes:
            self.scheduler.name_to_fused_node[snode.get_name()] = snode
        del self.scheduler.name_to_fused_node[self.get_name()]
        return self.scheduler.fuse_nodes(self.snodes)

    def add_fake_dep(self, fake_dep: Dep) -> None:
        # [FUNCTION DOC] Adds a fake read dependency and records it as unmet.
        self.set_read_writes(self.read_writes.with_read(fake_dep))
        self.unmet_dependencies.add(fake_dep)

    @cache_on_self
    def get_name(self) -> str:
        # [FUNCTION DOC] Cached. Name formed by joining member names with underscores.
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        # [FUNCTION DOC] Returns the name of the first member.
        return self.snodes[0].get_name()

    @cache_on_self
    def get_buffer_names(self) -> OrderedSet[str]:
        # [FUNCTION DOC] Cached. Union of member output buffer names.
        return OrderedSet.union(*[x.get_buffer_names() for x in self.snodes])

    def get_outputs(self) -> list[SchedulerBuffer]:
        # [FUNCTION DOC] Returns the concatenated outputs of all members.
        result: list[SchedulerBuffer] = []
        for node in self.snodes:
            result.extend(node.get_outputs())
        return result

    @cache_on_self
    def estimate_flops(self) -> int | None:
        # [FUNCTION DOC] Cached. Sum of FLOP estimates of template and extern members (None if none).
        # don't increment counters in fused methods so we don't double count
        fps = list(
            filter(
                None,
                (
                    node.estimate_flops()
                    for node in self.get_nodes()
                    if node.is_template() or node.is_extern()
                ),
            )
        )
        if len(fps) == 0:
            return None
        ret = sum(fps)
        return ret

    def get_nodes(self) -> Sequence[BaseSchedulerNode]:
        # [FUNCTION DOC] Returns the members.
        return self.snodes

    def get_device(self) -> torch.device | None:
        # [FUNCTION DOC] Returns the device of the first member, or None if there are no members.
        return self.snodes[0].get_device() if self.snodes else None

    @classmethod
    def can_fuse(cls, producer: BaseSchedulerNode, consumer: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] Always False: a grouped node cannot be fused with another node.
        # GroupedSchedulerNode cannot be fused with another node
        return False


def pick_loop_order(
    stride_lengths: list[list[int]],
    sizes: Sequence[sympy.Expr],
    priority_idx: Sequence[int] = (),
) -> list[int]:
    # [FUNCTION DOC] Heuristic loop iteration order. Given per-buffer stride lengths for each loop dimension and loop sizes, returns the
    # dimension indices ordered so that size-one dims go last and, otherwise, dims with smaller (absolute, non-zero) strides across more
    # buffers come first, ties broken toward contiguous (higher index first). When priority_idx is non-empty only those buffers' strides
    # are considered. If config.pick_loop_orders is off, returns the plain reversed index order.
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Comparator on dimension indices a and b used for sorting: size-one dims sort last; otherwise votes by counting
        # buffers where each dim has the smaller stride (zero stride counts as favorable to the other), returning -1, 1 or the contiguity tie break.
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        # Take abs, otherwise flipped dimensions are treated as smaller
        # strides than contiguous dims
        stride_len_a = [abs(sl[a]) for sl in stride_lengths]
        stride_len_b = [abs(sl[b]) for sl in stride_lengths]

        # equivalent to
        # np.logical_or(stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]).all()
        a_first = sum(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = sum(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first > b_first:
            return -1
        if b_first > a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        # if we have priority node, only use that node's order
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


def _replace_operation_buffer(
    orig_node: ir.MultiTemplateBuffer, new_node: ir.OperationBuffer
) -> None:
    # [FUNCTION DOC] HELPER: Replaces a MultiTemplateBuffer in the graph with the chosen operation buffer: the new node takes over the original
    # buffer and operation names and takes its position in the graph's buffers and operations lists and name maps (mutates V.graph).
    # Raises AssertionError if names are not strings.
    replaced_buf_name = new_node.get_name()
    orig_buf_name = orig_node.get_name()
    if not (isinstance(orig_buf_name, str) and isinstance(replaced_buf_name, str)):
        raise AssertionError("expected orig_buf_name and replaced_buf_name to be str")

    replaced_op_name = new_node.get_operation_name()
    orig_op_name = orig_node.get_operation_name()
    if not (isinstance(orig_op_name, str) and isinstance(replaced_op_name, str)):
        raise AssertionError("expected orig_op_name and replaced_op_name to be str")

    del V.graph.name_to_buffer[replaced_buf_name]
    new_node.name = orig_buf_name

    del V.graph.name_to_op[replaced_op_name]
    new_node.operation_name = orig_op_name

    orig = V.graph.buffers.index(orig_node)
    V.graph.buffers.remove(new_node)
    V.graph.buffers[orig] = new_node
    V.graph.name_to_buffer[orig_buf_name] = new_node

    orig = V.graph.operations.index(orig_node)
    V.graph.operations.remove(new_node)
    V.graph.operations[orig] = new_node
    V.graph.name_to_op[orig_op_name] = new_node


def _estimate_fused_epilogue_runtime(node1, node2, epilogue_runtime) -> float:
    # [FUNCTION DOC] HELPER: Estimates the extra runtime of fusing an epilogue into a template: epilogue runtime scaled by the saturating ratio of
    # extra bytes the epilogue reads (beyond the template's written bytes) to the total.
    # KNOWN ISSUE: divides by the template write bytes, which would fail if that were zero.
    template_write_bytes = node1.get_write_buffer_sizes()
    epilogue_read_bytes = node2.get_read_buffer_sizes()
    extra_bytes = epilogue_read_bytes - template_write_bytes
    # If no extra memory read by epilogue, assume epilogue is free
    # if extra memory is read by epilogue, add to minimum choice
    extra_bytes_ratio = extra_bytes / template_write_bytes

    # Smoothly approaches 1 as extra_bytes_ratio increases
    extra_memory_ratio = extra_bytes_ratio / (1 + extra_bytes_ratio)
    return extra_memory_ratio * epilogue_runtime


def _occupancy_before_and_after_fusion(
    unfused_n_regs: int,
    fused_n_regs: int,
    fused_n_spills: int,
    num_warps: int,
    device_props: DeviceProperties,
) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Returns (blocks per SM unfused, blocks per SM fused) from register counts, num_warps and device properties. Returns
    # (0, -1) if the fused kernel spills 8 or more registers (reject), and (1, 1) if device info is unavailable. Raises AssertionError
    # if num_warps is falsy.
    if fused_n_spills >= 8:
        return 0, -1

    # # Need device info to calculate occupancy
    regs_per_sm = device_props.regs_per_multiprocessor
    warp_size = device_props.warp_size
    if regs_per_sm is None or warp_size is None:
        return 1, 1  # Can't calculate, allow fusion

    if not num_warps:
        raise AssertionError("expected num_warps to be truthy")
    threads_per_block = num_warps * warp_size

    regs_per_block_unfused = unfused_n_regs * threads_per_block
    regs_per_block_fused = fused_n_regs * threads_per_block

    blocks_unfused = regs_per_sm // regs_per_block_unfused
    blocks_fused = regs_per_sm // regs_per_block_fused

    return blocks_unfused, blocks_fused


def _fuse_epilogue(
    ms1: float,
    ms2: float,
    unfused_n_regs: int,
    fused_n_regs: int,
    fused_n_spills: int,
    num_warps: int,
    device_props: DeviceProperties,
) -> bool:
    # [FUNCTION DOC] HELPER: Decides whether to fuse an epilogue into a GEMM template given runtimes of the template (ms1) and epilogue (ms2),
    # register use, spills, warps and device properties. Rejects on heavy spilling; accepts if fused occupancy is at least 4 blocks, the
    # occupancy ratio vs unfused exceeds one half, or the epilogue dominates runtime (more than twice) with occupancy above one block.
    """
    Determine whether to fuse an epilogue into a GEMM template.
    """
    MIN_ACCEPTED_OCCUPANCY = 4
    REGRESSED_OCCUPANCY_RATIO = 0.5

    # Check occupancy impact
    blocks_unfused, blocks_fused = _occupancy_before_and_after_fusion(
        unfused_n_regs, fused_n_regs, fused_n_spills, num_warps, device_props
    )

    epilogue_dominated_with_sufficient_occupancy = ms2 > 2 * ms1 and blocks_fused > 1

    # fuse if no major register spills
    # Occupancy can decrease but if memory bound/epilogue dominated
    # optimistically fuse
    return blocks_fused != -1 and (
        blocks_fused >= MIN_ACCEPTED_OCCUPANCY
        or blocks_fused / blocks_unfused > REGRESSED_OCCUPANCY_RATIO
        or epilogue_dominated_with_sufficient_occupancy
    )


@dataclasses.dataclass
class NodeUser:
    # [CLASS DOC] A user of a buffer: the user node (scheduler node or OutputNode), whether it can update the buffer in place, and whether it
    # is a weak user (must be scheduled after but does not read the value). Equality and hash use node name and both flags.
    node: BaseSchedulerNode | OutputNode
    can_inplace: bool = False

    # A weak user must be scheduled after a given node, but doesn't actually
    # use the result
    is_weak: bool = False

    def __hash__(self) -> int:
        # [FUNCTION DOC] Hash from the node name and flags.
        return hash((self.node.get_name(), self.can_inplace, self.is_weak))

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] Equal to another NodeUser with the same node name, in-place flag and weak flag.
        return (
            isinstance(other, NodeUser)
            and self.get_name() == other.get_name()
            and self.can_inplace == other.can_inplace
            and self.is_weak == other.is_weak
        )

    def get_name(self) -> str:
        # [FUNCTION DOC] Returns the name of the user node.
        return self.node.get_name()

    def merge(self, other: NodeUser) -> NodeUser:
        # [FUNCTION DOC] Merges with another NodeUser of the same node: in-place only if both allow it, weak only if both are weak. Raises
        # AssertionError if the nodes differ.
        if self.node is not other.node:
            raise AssertionError("expected self.node to be other.node")
        return NodeUser(
            self.node,
            self.can_inplace and other.can_inplace,
            self.is_weak and other.is_weak,
        )


_post_grad_graph_counter = itertools.count()


def get_layout_symints(node: ir.IRNode) -> OrderedSet[sympy.Symbol]:
    # [FUNCTION DOC] Returns the set of free symbols in a node's layout (size, stride, offset), recursing into the target of mutation layouts.
    # Raises AssertionError if the layout is neither a Layout nor None.
    """Get free symbols from a node's layout (size, stride, offset)."""
    free_symbol_uses: OrderedSet[sympy.Symbol] = OrderedSet()
    layout = node.maybe_get_layout()
    if isinstance(layout, ir.Layout):
        free_symbol_uses.update(
            free_symbols(layout.size)
            | free_symbols(layout.stride)
            | free_symbols(layout.offset)
        )
        if isinstance(layout, ir.MutationLayoutSHOULDREMOVE):
            # symint may be used as index in layout.target
            free_symbol_uses.update(get_layout_symints(layout.target))
    else:
        if layout is not None:
            raise AssertionError(f"Expect layout to be None but found layout={layout}")
    return free_symbol_uses


def get_scheduler_node_symbol_uses(
    node: BaseSchedulerNode,
) -> OrderedSet[sympy.Symbol]:
    # [FUNCTION DOC] Returns the symbols used by a scheduler node: for fused nodes the union over sub-nodes; otherwise the IR node's free symbol
    # uses plus layout symints of its outputs. Raises AssertionError if the node has no IR node.
    """
    Gets symbols used in a scheduler node, including free symbols from
    the node's operations and layout symints from outputs.
    """
    if isinstance(node, FusedSchedulerNode):
        return OrderedSet().union(
            *(get_scheduler_node_symbol_uses(snode) for snode in node.snodes)
        )
    if node.node is None:
        raise AssertionError("expected node.node to be set")
    free_symbol_uses = node.node.get_free_symbol_uses()
    free_symbol_uses.update(
        *(get_layout_symints(ir_node) for ir_node in node.node.get_outputs())
    )
    return free_symbol_uses


def _is_epilogue_fusion_enabled(template_node: BaseSchedulerNode) -> bool:
    # [FUNCTION DOC] HELPER: True if epilogue fusion is enabled for the template node: its per-template flag if set, else global config.
    """Check per-template flag, fall back to global config."""
    tb = template_node.get_template_node()
    if tb is not None and tb.allow_epilogue_fusion is not None:
        return tb.allow_epilogue_fusion
    return config.epilogue_fusion


def _is_atomic_add_mutation_epilogue(
    node: BaseSchedulerNode, *, check_config: bool = True
) -> bool:
    # [FUNCTION DOC] HELPER: True if node is a single-output scatter with atomic_add mode (all writes atomic add MemoryDeps) that mutates exactly one
    # buffer without aliasing; optionally requires the epilogue_fusion_with_atomic_add config.
    if check_config and not config.epilogue_fusion_with_atomic_add:
        return False
    if not isinstance(node, SchedulerNode):
        return False
    if not isinstance(node.node, ir.ComputedBuffer):
        return False
    if not isinstance(node.node.data, ir.Scatter):
        return False
    if node.node.data.scatter_mode != "atomic_add":
        return False

    outputs = node.get_outputs()
    if len(outputs) != 1:
        return False
    if outputs[0].get_aliases() or len(outputs[0].get_mutations()) != 1:
        return False

    writes = node.read_writes.writes
    return bool(writes) and all(
        isinstance(write, MemoryDep) and write.mode == "atomic_add" for write in writes
    )


def _can_fuse_atomic_add_template_epilogue(
    template_node: BaseSchedulerNode, epilogue_node: BaseSchedulerNode
) -> bool:
    # [FUNCTION DOC] HELPER: True if the template is a Triton template buffer and the epilogue is an atomic add mutation epilogue.
    template_buf = template_node.get_template_node()
    return isinstance(
        template_buf, ir.TritonTemplateBuffer
    ) and _is_atomic_add_mutation_epilogue(epilogue_node)


def _is_prologue_fusion_enabled(template_node: BaseSchedulerNode) -> bool:
    # [FUNCTION DOC] HELPER: True if prologue fusion is enabled for the template node: its per-template flag if set, else global config.
    """Check per-template flag, fall back to global config."""
    tb = template_node.get_template_node()
    if tb is not None and tb.allow_prologue_fusion is not None:
        return tb.allow_prologue_fusion
    return config.prologue_fusion


def is_epilogue_fusion(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    # [FUNCTION DOC] True if node1 is a template, node2 is not, and epilogue fusion is enabled for node1.
    return (
        node1.is_template()
        and not node2.is_template()
        and _is_epilogue_fusion_enabled(node1)
    )


def is_prologue_fusion(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    # [FUNCTION DOC] True if node2 is a template, node1 is not, and prologue fusion is enabled for node2.
    return (
        node2.is_template()
        and not node1.is_template()
        and _is_prologue_fusion_enabled(node2)
    )


def is_template_fusion(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    # [FUNCTION DOC] True if the pair is an epilogue or prologue fusion with a template.
    return is_epilogue_fusion(node1, node2) or is_prologue_fusion(node1, node2)


def template_fusion_pw_node(node1: BaseSchedulerNode, node2: BaseSchedulerNode):
    # [FUNCTION DOC] Returns the non-template (pointwise) node of a template fusion pair: node2 for epilogue fusion, else node1.
    return node2 if is_epilogue_fusion(node1, node2) else node1


def _iter_loop_state_nodes(
    nodes: Iterable[BaseSchedulerNode],
) -> Iterator[SchedulerNode | FusedSchedulerNode]:
    # [FUNCTION DOC] HELPER: Generator over nodes yielding leaf SchedulerNodes and FusedSchedulerNodes (children first, then the fused node), skipping other node kinds.
    for node in nodes:
        if isinstance(node, FusedSchedulerNode):
            yield from _iter_loop_state_nodes(node.snodes)
            yield node
        elif isinstance(node, SchedulerNode):
            yield node


@dataclasses.dataclass
class _LoopStateSnapshot:
    # [CLASS DOC] HELPER: Snapshot of loop state for a set of nodes: per-leaf-SchedulerNode state tuples and per-FusedSchedulerNode group
    # values, so that speculative loop mutations can be rolled back via restore.
    """Captured loop state for a set of scheduler nodes, restorable on rollback.

    Stores both SchedulerNode loop state (body, sizes, deps) and
    FusedSchedulerNode group assignments, since the latter is reassigned
    directly after child reindexing and has no mutation listener.
    """

    scheduler_node_states: dict[SchedulerNode, tuple[Any, ...]] = dataclasses.field(
        default_factory=dict
    )
    fused_node_groups: dict[FusedSchedulerNode, Any] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def create(cls, nodes: tuple[BaseSchedulerNode, ...]) -> _LoopStateSnapshot:
        # [FUNCTION DOC] HELPER: Class method. Builds a snapshot of all leaf and fused nodes reachable from nodes.
        """Capture scheduler-node boundaries and their mutable leaf loop state."""
        snapshot = cls()
        for node in _iter_loop_state_nodes(nodes):
            if isinstance(node, FusedSchedulerNode):
                snapshot._snapshot_fused_node(node)
            else:
                snapshot._snapshot_scheduler_node(node)
        return snapshot

    def _snapshot_scheduler_node(self, sn: SchedulerNode) -> None:
        # [FUNCTION DOC] HELPER: Records the loop state of a leaf node; raises AssertionError if already recorded.
        """Capture one leaf scheduler node before its first loop mutation."""
        if sn in self.scheduler_node_states:
            raise AssertionError(f"scheduler node {sn} already snapshotted")
        self.scheduler_node_states[sn] = sn.snapshot_loop_state()

    def _snapshot_fused_node(self, node: FusedSchedulerNode) -> None:
        # [FUNCTION DOC] HELPER: Records the group of a fused node; raises AssertionError if already recorded.
        """Capture fused-node group metadata changed outside leaf listeners."""
        if node in self.fused_node_groups:
            raise AssertionError(f"fused node {node} already snapshotted")
        self.fused_node_groups[node] = node.group

    def restore(self) -> None:
        # [FUNCTION DOC] HELPER: Restores leaf loop states and fused node groups (refreshing fused node dependencies).
        """Restore all captured loop state and fused-node group metadata."""
        for sn, state in self.scheduler_node_states.items():
            sn.restore_loop_state(state)
        for node, group in self.fused_node_groups.items():
            node.group = group
            refresh_group_node_dependencies(node)


@dataclasses.dataclass
class _LoopMutationTracker:
    # [CLASS DOC] HELPER: Rollback scope for speculative loop reordering/reindexing during can_fuse. Installs itself as loop mutation
    # listener on leaf nodes (chaining previous listeners), lazily snapshots the original state on first mutation, and on finish detaches
    # and optionally restores the state. State: candidate nodes, watched nodes, previous listeners, snapshot.
    """Rollback scope for speculative loop mutations during can_fuse().

    can_fuse() may speculatively reorder or reindex loops while evaluating
    whether a fusion is legal. If the final decision rejects the fusion,
    this tracker restores the original loop structure so later fusion
    candidates do not inherit a speculative layout chosen for a fusion
    that did not happen.

    Recursive can_fuse() calls chain their listeners so each scope captures its
    own decision boundary while the outer scope still sees nested mutations.

    Use finish(rollback=False) to keep mutations or finish(rollback=True) to
    restore the original state. If no mutation occurred, finish() is a no-op.
    """

    nodes: tuple[BaseSchedulerNode, ...]
    watched_nodes: OrderedSet[SchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    previous_listeners: dict[SchedulerNode, Callable[[SchedulerNode], None] | None] = (
        dataclasses.field(default_factory=dict)
    )
    state: _LoopStateSnapshot | None = None

    @classmethod
    def create(cls, nodes: tuple[BaseSchedulerNode, ...]) -> _LoopMutationTracker:
        # [FUNCTION DOC] HELPER: Class method. Creates a tracker over the deduplicated nodes and watches every leaf scheduler node.
        """Create a rollback scope and watch mutable leaf scheduler nodes."""
        seen = OrderedSet(nodes)
        tracker = cls(nodes=tuple(seen))
        for node in _iter_loop_state_nodes(seen):
            if isinstance(node, SchedulerNode):
                tracker.watch(node)
        return tracker

    def watch(self, sn: SchedulerNode) -> None:
        # [FUNCTION DOC] HELPER: Installs this tracker as the mutation listener of leaf node sn (saving the previous listener); no-op if watched.
        """Install this scope as the mutation listener for a leaf node."""
        if sn in self.watched_nodes:
            return
        self.previous_listeners[sn] = sn._loop_mutation_listener
        self.watched_nodes.add(sn)
        sn._loop_mutation_listener = self.track

    def track(self, sn: SchedulerNode) -> None:
        # [FUNCTION DOC] HELPER: Mutation listener: forwards to the previous listener and, on the first mutation, snapshots the original loop
        # state of the candidate nodes. Raises AssertionError for unwatched nodes.
        """Lazily snapshot candidate roots when the first mutation occurs."""
        if sn not in self.watched_nodes:
            raise AssertionError(f"scheduler node {sn} is not being watched")
        if previous := self.previous_listeners[sn]:
            previous(sn)
        if self.state is not None:
            # Keep the original pre-mutation snapshot for the whole scope.
            return

        # The listener tells us a child loop mutated. Snapshot the original
        # candidate roots here so we also capture fused-node group state,
        # which is reassigned directly and has no listener of its own.
        self.state = _LoopStateSnapshot.create(self.nodes)

    def finish(self, *, rollback: bool) -> None:
        # [FUNCTION DOC] HELPER: Restores the previous listeners on all watched nodes; if rollback is requested and a snapshot exists, restores the
        # loop state.
        """Detach listeners and restore captured state if rolling back."""
        for sn in self.watched_nodes:
            sn._loop_mutation_listener = self.previous_listeners[sn]
        if not rollback or self.state is None:
            return
        self.state.restore()


# Distinguishes "not cached" from a cached None in _tiling_memory_cache.
_TILING_MEMORY_MISS = object()


class Scheduler:
    # [CLASS DOC] Graph of scheduler nodes built from the lowered IR operations. Responsible for dependency computation, ordering,
    # dead node elimination, fusion (including template, mix-order, nested reduction, foreach/combo), loop merging, memory and
    # communication reordering, stream/mempool assignment, graph partitioning, and final code generation. Key state: nodes, name
    # to node/buffer/fused node maps, backends per device, mutation renames, completed operations, available buffer names.
    """
    A Scheduler is a graph of BaseSchedulerNodes. It is responsible for
    optimizations such as fusion, reorder, and graph partition.
    """

    def __init__(self, nodes: list[ir.Operation]) -> None:
        # [FUNCTION DOC] Constructs the scheduler, running the full initialization pipeline under a timing context.
        with dynamo_timed("Scheduler.__init__"):
            self._init(nodes)

    @staticmethod
    def count_kernel_nodes(nodes: Sequence[BaseSchedulerNode]) -> int:
        # [FUNCTION DOC] Static. Counts the nodes that are not no-op kernel nodes.
        return sum(1 for node in nodes if not isinstance(node, NopKernelSchedulerNode))

    def _init(self, nodes: list[ir.Operation]) -> None:
        # [FUNCTION DOC] HELPER: Real constructor body. Registers self as V.graph.scheduler and builds scheduler nodes from the IR operations; sets up
        # name maps (nodes, buffers, donated buffers, fused nodes), mutation rename maps, then runs the pipeline: global comm ordering,
        # dependency computation, topological sort, dead node elimination, ancestors and input distances, pre-fusion custom pass, foreach
        # nodes, optional distributed autotune, stream and mempool assignment, fusion, post-fusion custom pass, loop merging, multi-template
        # finalization, switch ordering, then (per config) peak-memory reordering, combo kernels, compute/comm overlap passes, grouped node
        # unpacking, graph partition reordering, last-usage computation, memory check nodes, debug logging. Finally initializes codegen
        # state (buffers to free, origin index, stream context, removed ops) and records a graph stats metric row. Mutates global graph state
        # and metrics.
        self._tiling_memory_cache: dict[tuple[Any, ...], Any] = {}
        super().__init__()
        V.graph.scheduler = self
        self.backends: dict[torch.device, BaseScheduling] = {}
        self.post_grad_graph_id = next(_post_grad_graph_counter)
        self._graph_partition_counter = itertools.count()
        self.completed_operations: OrderedSet[str] = OrderedSet()
        self.available_buffer_names = OrderedSet(
            [
                *V.graph.graph_inputs.keys(),
                *V.graph.constants.keys(),
                *V.graph.torchbind_constants.keys(),
            ]
        )
        self.nodes = [self.create_scheduler_node(n) for n in nodes]
        self.previous_node: BaseSchedulerNode | None = None
        self.current_node: BaseSchedulerNode | None = None
        self.update_zero_dim_cpu_tensor()
        # some new constants could have been created above
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()

        # See [Note: Graph Partition Device Contexts]
        self.default_device_context: torch.device | None = None

        self.name_to_donated_buffer: dict[str, SchedulerDonatedBuffer] = (
            self.get_donated_buffers()
        )
        self.name_to_node: dict[str, BaseSchedulerNode] = {
            n.get_name(): n for n in self.nodes
        }

        self.name_to_buf: dict[str, SchedulerBuffer] = {
            buf.get_name(): buf for node in self.nodes for buf in node.get_outputs()
        }
        self.name_to_fused_node: dict[str, BaseSchedulerNode] = self.name_to_node.copy()

        # mutation_real_name: Maps back to the original name for codegen
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_real_name = {"buf0" : "buf1"}
        # all subsequent uses of buf0 become buf1's usage in dependency graph
        self.mutation_real_name: dict[str, str] = {}

        # We handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        # Example:
        # If you mutate buf0 inside of buf1's kernel, then:
        # mutation_renames = {"buf1" : "buf0"}
        # in codegen we only use buf0, never buf1
        self.mutation_renames: dict[str, str] = {}

        self.seen_template_fusions: OrderedSet[
            tuple[BaseSchedulerNode, BaseSchedulerNode]
        ] = OrderedSet()
        # Must run first to correctly set dependencies, before all other passes that rely on
        # reading from .read_writes.reads or .unmet_dependencies
        self.nodes = comms.decide_global_ordering_of_comms(
            self.nodes,
            self.name_to_buf,
            self.name_to_fused_node,
        )

        self.compute_dependencies()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.dead_node_elimination()
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.compute_ancestors()
        self.compute_input_distances()

        # pyrefly: ignore [bad-assignment]
        metrics.ir_nodes_pre_fusion += len(self.nodes)
        from torch._inductor.debug import log_ir_post_fusion, log_ir_pre_fusion

        log_ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.create_foreach_nodes()
        self.nodes = self.topological_sort_schedule(self.nodes)
        self.logged_slow_fusion = OrderedSet[tuple[str, str]]()
        if config._pre_fusion_custom_pass is not None:
            self.nodes = config._pre_fusion_custom_pass(self.nodes)

        # Distributed autotune can replace MultiTemplateBuffer nodes before the
        # normal mempool assignment pass runs. Initialize these maps now so
        # replacement can preserve any IR-level mempool metadata it sees.
        self.node_to_mempool: dict[BaseSchedulerNode, tuple[int, int] | None] = {}
        self.buff_to_mempool: dict[str, tuple[int, int] | None] = {}
        self._mempool_nodes: bool = False

        if config.distributed_max_autotune_gemm:
            from . import distributed_autotune

            distributed_autotune.schedule(self)
            self.compute_ancestors()

        # Stream assignments must be populated BEFORE fusion
        # to prevent fusing nodes across stream boundaries
        self.node_to_stream: dict[BaseSchedulerNode, int] = {}
        self.buff_to_stream: dict[str, int] = {}
        self._multi_stream_nodes: bool = False
        # Maps stream_idx → user_object_index for retrieving user stream objects
        self.stream_idx_to_user_obj_idx: dict[int, int] = {}
        self.user_obj_idx_to_stream_idx: dict[int, int] = {}
        self._populate_stream_assignments()
        self._populate_mempool_assignments()

        self.nodes = self.fuse_nodes(self.nodes)
        if config._post_fusion_custom_pass is not None:
            self.nodes = config._post_fusion_custom_pass(self.nodes)

        if any(
            isinstance(node, FusedExternTritonKernelSchedulerNode)
            for node in self.nodes
        ):
            # if a user triton kernel has been epilogue-fused,
            # there is likely an opportunity to prune an NopKernel
            # (which is originally used to generate the buffer which the triton kernel writes to)
            self.dead_node_elimination()

        self.merge_loops()
        self.finalize_multi_template_buffers()
        if (
            config.max_autotune_gemm or config.max_autotune
        ) and use_pipelined_autotuning():
            torch._inductor.select_algorithm.PrecompileThreadPool.shutdown_instance()

        # torch.cond and torch.switch can contain arbitrary subgraphs with collectives;
        # reordering them can cause an nccl hang.
        self._enforce_switch_ordering()

        # Peak memory pass and overlap pass must run last, otherwise
        # other reordering passes could undo their effects.
        if config.reorder_for_peak_memory:
            from .memory import reorder_for_peak_memory

            self.nodes = reorder_for_peak_memory(
                self.nodes,
                self.name_to_buf,
                self.name_to_fused_node,
                OrderedSet(V.graph.graph_inputs.keys()),
                OrderedSet(V.graph.get_output_names()),
            )

        if config.combo_kernels:
            with dynamo_timed(
                "Scheduler.create_combo_kernel_nodes",
                log_pt2_compile_event=True,
                log_waitcounter=True,
            ):
                self.create_combo_kernel_nodes(num_ck_nodes=None)
            from .memory import assign_memory_planning_info_for_scheduler_buffers

            assign_memory_planning_info_for_scheduler_buffers(
                self.nodes, self.name_to_buf
            )

        # reorder_for_compute_comm_overlap may do benchmarking to estimate
        # op runtime. Disable it for now in deterministic mode.
        if not config.deterministic and config.reorder_for_compute_comm_overlap:
            if not config.reorder_for_peak_memory:
                from .memory import assign_memory_planning_info_for_scheduler_buffers

                assign_memory_planning_info_for_scheduler_buffers(
                    self.nodes, self.name_to_buf
                )

            if config_comms.runtime_estimations_align_across_all_distributed_ranks:
                has_collectives = False
                for node in self.nodes:
                    if is_collective(node.node):
                        has_collectives = True
                        break
                if has_collectives:
                    from .comms import (
                        align_runtime_estimations_across_all_distributed_ranks,
                    )

                    align_runtime_estimations_across_all_distributed_ranks(self.nodes)

            # pyrefly: ignore [unbound-name]
            if config_comms.reorder_sink_verbose_logging:
                from torch._logging import trace_structured

                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "scheduler_nodes_before_comm_overlap",
                        "encoding": "string",
                    },
                    payload_fn=lambda: "\n\n".join(
                        [
                            f"snode[{i}]"
                            + n.debug_str()
                            + f" buffer_names:{n.get_buffer_names()}"
                            for i, n in enumerate(self.nodes)
                        ]
                    ),
                )
            self.nodes = comms.reorder_compute_and_comm_for_overlap(self.nodes)

        if config.aten_distributed_optimizations.enable_simple_overlap:
            if (
                not config.reorder_for_peak_memory
                and not config.reorder_for_compute_comm_overlap
            ):
                from .memory import assign_memory_planning_info_for_scheduler_buffers

                assign_memory_planning_info_for_scheduler_buffers(
                    self.nodes, self.name_to_buf
                )
            with dynamo_timed(
                "Scheduler.simple_overlap",
                log_pt2_compile_event=True,
                log_waitcounter=True,
            ):
                self.nodes = comms.simple_overlap(self.nodes)

        self.process_grouped_nodes()

        if (
            # pyrefly: ignore[unbound-name]
            config.graph_partition
            # pyrefly: ignore[unbound-name]
            and config.triton.cudagraphs
            # pyrefly: ignore[unbound-name]
            and config.triton.reorder_for_reducing_graph_partitions
        ):
            self.nodes = self.maybe_reorder_for_minimizing_partition(self.nodes)
            self.nodes = self.reorder_for_partition_with_simple_dependency(self.nodes)

        self.compute_last_usage()

        if torch._inductor.config.test_configs.track_memory_lifecycle:
            self.insert_memory_check_nodes()

        log_ir_post_fusion(self.nodes)
        # pyrefly: ignore[unbound-name]
        V.debug.graph_diagram(self.nodes)
        self.debug_draw_graph()

        # used during codegen:
        self.buffer_names_to_free: OrderedSet[str] = OrderedSet()

        # fx graph node to the position it appears in the graph
        # for debug attribution
        self.origin_to_index: dict[torch.fx.Node, int] = {}

        # The only source of which stream context we are currently in during the codegen phase.
        self._current_stream_ctx: EnterCudaStreamContextLine | None = None

        get_metric_table("graph_stats").add_row(
            lambda: {
                "graph_id": self.post_grad_graph_id,
                "num_nodes_before_fusion": self.num_orig_nodes,
                "num_nodes_after_fusion": len(self.nodes),
            }
        )

        # Unlike V.graph.removed_buffers, the op recorded here is removed but
        # we still need the buffer (generated in alternative ways)
        self.removed_ops: OrderedSet[str] = OrderedSet()

    def get_donated_buffers(self) -> dict[str, SchedulerDonatedBuffer]:
        # [FUNCTION DOC] Builds a mapping from name to SchedulerDonatedBuffer for each donated buffer among the graph's original inputs.
        name_to_donated_buf = {}
        for name in V.graph.graph_inputs_original:
            if isinstance(V.graph.graph_inputs_original[name], ir.DonatedBuffer):
                name_to_donated_buf[name] = SchedulerDonatedBuffer(
                    self,
                    V.graph.graph_inputs_original[name],
                    defining_op=None,
                )
        return name_to_donated_buf

    def _populate_stream_assignments(self) -> None:
        # [FUNCTION DOC] HELPER: Fills node_to_stream and buff_to_stream from each IR node's stream index (default stream index when none), mapping
        # user stream objects to dense stream indexes. If any non-default stream exists, gives device-less none-layout buffers a device so
        # they enter device guards, and sets the multi-stream flag. Mutates IR layouts.
        """Populate node_to_stream and buff_to_stream from IR node stream_idx.

        Reads the stream_idx field set on IR nodes during lowering to determine
        which stream each scheduler node should run on. This field is propagated
        from 'custom.stream' FX node metadata via IRNode.current_stream_idx().
        """
        from .stream_constants import DEFAULT_STREAM_IDX

        for node in self.nodes:
            stream_idx = DEFAULT_STREAM_IDX

            if node.node is not None:
                user_obj_idx = node.node.get_stream_idx()
                if user_obj_idx is not None:
                    stream_idx = self._get_or_create_stream_idx_for_user_obj(
                        user_obj_idx
                    )

            self.node_to_stream[node] = stream_idx

            # Also populate buff_to_stream for all buffers produced by this node.
            # Mutation renames are resolved at lookup time via get_buf_stream.
            for buf in node.get_buffer_names():
                self.buff_to_stream[buf] = stream_idx

        # Propagate a device to device-less nodes (e.g. record_event,
        # wait_event) so they naturally enter the device guard in the
        # main codegen loop instead of requiring special-case handling.
        if any(s != DEFAULT_STREAM_IDX for s in self.node_to_stream.values()):
            device = next(
                (n.get_device() for n in self.nodes if n.get_device() is not None), None
            )
            if device is not None:
                for node in self.nodes:
                    ir_node = node.node
                    if (
                        node.get_device() is None
                        and isinstance(ir_node, ir.Buffer)
                        and isinstance(ir_node.layout, ir.NoneLayout)
                    ):
                        # pyrefly: ignore [bad-assignment]
                        ir_node.layout = ir.NoneLayout(device=device)

        # Check if we have any nodes on non-default streams
        self._multi_stream_nodes = any(
            stream_idx != DEFAULT_STREAM_IDX
            for stream_idx in self.node_to_stream.values()
        )

    def _get_or_create_stream_idx_for_user_obj(self, user_obj_idx: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the dense stream index for a user stream object index, allocating the next index (starting at 1; 0 is the
        # default stream) and recording the two-way mapping on first use.
        stream_idx = self.user_obj_idx_to_stream_idx.get(user_obj_idx)
        if stream_idx is None:
            # 0 is reserved for the default stream.
            stream_idx = len(self.user_obj_idx_to_stream_idx) + 1
            self.user_obj_idx_to_stream_idx[user_obj_idx] = stream_idx
            self.stream_idx_to_user_obj_idx[stream_idx] = user_obj_idx
        return stream_idx

    def get_node_stream(self, node: BaseSchedulerNode) -> int:
        # [FUNCTION DOC] Returns the stream index of a node, computing and caching it if missing: from the single stream of its sub-nodes, else its IR
        # stream, else the stream of its buffers, else the default. Raises AssertionError if sub-nodes or buffers span multiple streams.
        # Side effects: updates node_to_stream, buff_to_stream and the multi-stream flag.
        from .stream_constants import DEFAULT_STREAM_IDX

        if node in self.node_to_stream:
            return self.node_to_stream[node]

        child_streams = OrderedSet(
            self.get_node_stream(child)
            for child in node.get_nodes()
            if child is not node
        )
        if len(child_streams) == 1:
            stream_idx = next(iter(child_streams))
        elif len(child_streams) > 1:
            raise AssertionError(
                f"Scheduler node {node.get_name()} combines multiple streams: "
                f"{list(child_streams)}"
            )
        elif (
            node.node is not None
            and (user_obj_idx := node.node.get_stream_idx()) is not None
        ):
            stream_idx = self._get_or_create_stream_idx_for_user_obj(user_obj_idx)
        else:
            buffer_streams = OrderedSet(
                self.buff_to_stream[buf_name]
                for buf_name in node.get_buffer_names()
                if buf_name in self.buff_to_stream
            )
            if len(buffer_streams) > 1:
                raise AssertionError(
                    f"Scheduler node {node.get_name()} has outputs on multiple "
                    f"streams: {list(buffer_streams)}"
                )
            stream_idx = (
                next(iter(buffer_streams)) if buffer_streams else DEFAULT_STREAM_IDX
            )

        self.node_to_stream[node] = stream_idx
        if stream_idx != DEFAULT_STREAM_IDX:
            self._multi_stream_nodes = True
        for buf_name in node.get_buffer_names():
            self.buff_to_stream.setdefault(buf_name, stream_idx)
        return stream_idx

    def _has_multi_stream_nodes(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether any node was assigned to a non-default stream.
        """Check if any nodes are assigned to non-default streams."""
        return self._multi_stream_nodes

    def _populate_mempool_assignments(self) -> None:
        # [FUNCTION DOC] HELPER: Rebuilds node_to_mempool and buff_to_mempool from IR metadata for all nodes and sets the flag telling whether any
        # user mempool is used.
        """Populate node_to_mempool and buff_to_mempool from IR node metadata."""
        self.node_to_mempool.clear()
        self.buff_to_mempool.clear()
        for node in self.nodes:
            mempool = self._get_node_mempool(node)
            self.node_to_mempool[node] = mempool

            # Also populate buff_to_mempool for all buffers produced by this node.
            # Mutation renames are resolved at lookup time via get_buf_mempool.
            for buf in node.get_buffer_names():
                self.buff_to_mempool[buf] = mempool

        self._mempool_nodes = any(
            mempool is not None for mempool in self.node_to_mempool.values()
        )

    def _has_mempool_nodes(self) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether any node is assigned to a user CUDA MemPool.
        """Check if any nodes are assigned to a user CUDA MemPool."""
        return self._mempool_nodes

    def _get_node_mempool(self, node: BaseSchedulerNode) -> tuple[int, int] | None:
        # [FUNCTION DOC] HELPER: Returns the mempool of a node: from its IR node, or for grouped nodes the single common mempool of the sub-nodes
        # (None if none). Raises AssertionError if sub-nodes mix mempools.
        """Return the CUDA MemPool assigned to a scheduler node.

        Grouped scheduler nodes such as ForeachKernelSchedulerNode have
        node.node == None, so derive their context from the underlying nodes.
        """
        if node.node is not None:
            return node.node.get_mempool()

        mempools = OrderedSet(
            self._get_node_mempool(subnode)
            for subnode in node.get_nodes()
            if subnode is not node
        )
        if not mempools:
            return None
        if len(mempools) != 1:
            raise AssertionError(
                f"Scheduler node {node.get_name()} contains mixed CUDA MemPool "
                f"contexts: {list(mempools)}"
            )
        return next(iter(mempools))

    def get_buf_stream(self, buf_name: str) -> int:
        # [FUNCTION DOC] Returns the stream index of a buffer, resolving mutation renames; falls back to the unrenamed name, then the default stream 0.
        """Return the stream index for a buffer, resolving mutation renames."""
        real = self.mutation_renames.get(buf_name, buf_name)
        return self.buff_to_stream.get(real, self.buff_to_stream.get(buf_name, 0))

    def get_buf_mempool(self, buf_name: str) -> tuple[int, int] | None:
        # [FUNCTION DOC] Returns the mempool of a buffer, resolving mutation renames; None if unknown.
        """Return the CUDA MemPool for a buffer, resolving mutation renames."""
        real = self.mutation_renames.get(buf_name, buf_name)
        return self.buff_to_mempool.get(real, self.buff_to_mempool.get(buf_name, None))

    def has_cross_stream_hazard(self, buf_name: str, node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] True if multiple streams are in use and the buffer was produced on a different stream than the node runs on.
        """True if buf_name was produced on a different stream than node.

        Resolves mutation renames so that mutated buffers inherit the
        stream of their original definition.
        """
        if not self._has_multi_stream_nodes():
            return False
        return self.get_buf_stream(buf_name) != self.get_node_stream(node)

    @property
    def current_device(self) -> torch.device | None:
        # [FUNCTION DOC] Property: the current device context, stored on V.graph.
        return V.graph.current_device

    @current_device.setter
    def current_device(self, device: torch.device | None) -> None:
        # [FUNCTION DOC] Setter: sets the current device context on V.graph.
        V.graph.current_device = device

    def debug_draw_graph(self) -> None:
        # [FUNCTION DOC] When environment variable INDUCTOR_WRITE_SCHEDULER_GRAPH is 1, draws the scheduler graph to an image (side effect on disk).
        """Generate an image of the graph for debugging"""
        if os.environ.get("INDUCTOR_WRITE_SCHEDULER_GRAPH", None) == "1":
            from .debug import draw_buffers

            draw_buffers(self.nodes, print_graph=True)

    def debug_print_nodes(self, label: str) -> None:
        # [FUNCTION DOC] When info logging is enabled, logs a label and the details of every node.
        if log.isEnabledFor(logging.INFO):
            log.info("%s:", label)
            for node in self.nodes:
                node.log_details()

    def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode:
        # [FUNCTION DOC] Wraps an IR operation in the right scheduler node: no-op nodes, SchedulerNode for computed/template buffers, extern kernel
        # node for ExternKernel. Raises AssertionError if the operation has no origins and NotImplementedError for other kinds.
        if node.get_origins() is None:
            raise AssertionError("All nodes passed to scheduling must have an origin")
        if node.is_no_op():
            return NopKernelSchedulerNode(self, node)
        elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
            return SchedulerNode(self, node)
        elif isinstance(node, ir.ExternKernel):
            return ExternKernelSchedulerNode(self, node)
        else:
            raise NotImplementedError(node)

    def create_foreach_nodes(self) -> None:
        # [FUNCTION DOC] Builds a ForeachKernelSchedulerNode for each foreach list recorded in the graph (ignoring eliminated and no-op nodes), points
        # the member names at it in the fused-node map, and replaces the members in self.nodes by the new foreach nodes (appended at the end).
        removed_node_names: OrderedSet[str] = OrderedSet()
        fe_nodes = []
        kept_node_names = self.name_to_fused_node.keys()

        for names in V.graph.lists.values():
            names = [
                name
                for name in names
                if name in kept_node_names
                and not isinstance(self.name_to_node[name], NopKernelSchedulerNode)
            ]
            if not names:
                # All nodes eliminated
                continue

            removed_node_names.update(names)
            snodes = [self.name_to_node[name] for name in names]

            enable_autotune = config.combo_kernels_autotune > 1
            fe_node = ForeachKernelSchedulerNode(
                self,
                snodes,
                use_custom_partition_algo=False,
                enable_autotune=enable_autotune,
            )

            fe_nodes.append(fe_node)

            for name in names:
                self.name_to_fused_node[name] = fe_node

        self.nodes = [
            node for node in self.nodes if node.get_name() not in removed_node_names
        ] + list(fe_nodes)

    def compute_dependencies(self) -> None:
        # [FUNCTION DOC] Computes users of every buffer and dependency edges between nodes. Builds, for each buffer name, a deduplicated list of NodeUser
        # (sharing lists between aliasing buffers), handles mutation by renaming buffers and adding fake dependencies (StarDep to the prior writer,
        # weak dependencies on prior readers, fake only for non-views), additional buffer/star dependencies, unbacked symbol def/use edges,
        # normal read edges, graph outputs, mutated inputs and constants as OutputNode users. Then sets users on buffers and donated buffers
        # and logs the user list. Mutates node dependencies, self.mutation_renames, self.mutation_real_name and V.graph.mutated_inputs.
        # KNOWN ISSUE: a local variable named str shadows the builtin in the final logging part (harmless there).
        """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """

        class DedupList(Generic[_T]):
            # [CLASS DOC] HELPER: List-like container that keeps elements unique (membership set alongside an item list) while still allowing append during iteration.
            """
            This data structure behaves like a list except it makes sure the
            elements remain unique.
            Normally one could use a OrderedSet/dict for this purpose however
            the list in question gets elements appended as it is being
            iterated over which means that we need to keep the list
            semantics.
            """

            def __init__(
                self,
                items: list[_T] | None = None,
                membership: OrderedSet[_T] | None = None,
            ) -> None:
                # [FUNCTION DOC] HELPER: Stores the items list and membership set (empty defaults).
                self.items = items or []
                self.membership = membership or OrderedSet()

            def append(self, node_user: _T) -> None:
                # [FUNCTION DOC] HELPER: Appends the item unless already present.
                if node_user in self.membership:
                    return
                self.items.append(node_user)
                self.membership.add(node_user)

            def __add__(self, other: DedupList[_T]) -> DedupList[_T]:
                # [FUNCTION DOC] HELPER: Returns a new deduplicated list with this list's items followed by other's items not in this list.
                new_membership = OrderedSet.union(self.membership, other.membership)
                new_items = self.items + [
                    x for x in other.items if x not in self.membership
                ]
                return DedupList(new_items, new_membership)

        # pyrefly: ignore [not-a-type]
        name_to_users: defaultdict[str, DedupList[NodeUser]] = collections.defaultdict(
            DedupList
        )

        # handle aliasing by using python aliasing in name_to_users
        # if foo aliases bar then we will make name_to_users["foo"] point
        # to the same python list as name_to_users["bar"]
        for node in self.nodes:
            for buf1 in node.get_outputs():
                buf1_name = buf1.get_name()
                # This is for handling auto functionized ops which return None
                # and mutate more than 1 inputs, we shouldn't let them all
                # point to the same user list since buffers in the aliases
                # list might not be alias to each other.
                if (
                    isinstance(buf1.node.layout, ir.NoneLayout)
                    and len(buf1.get_aliases()) > 1
                ):
                    continue
                for buf2_name in buf1.get_aliases():
                    if buf1_name in name_to_users and buf2_name in name_to_users:
                        # merge the two
                        list1 = name_to_users[buf1_name]
                        list2 = name_to_users[buf2_name]
                        combined = list1 + list2
                        for key in name_to_users:
                            if (
                                name_to_users[key] is list1
                                or name_to_users[key] is list2
                            ):
                                name_to_users[key] = combined
                    elif buf1_name in name_to_users:
                        name_to_users[buf2_name] = name_to_users[buf1_name]
                    else:
                        name_to_users[buf1_name] = name_to_users[buf2_name]

        # pyrefly: ignore [not-a-type]
        def rename(n: str) -> str:
            # [FUNCTION DOC] HELPER: Follows the chain of mutation renames from n and returns the final buffer name.
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def add_user(
            # pyrefly: ignore [not-a-type]
            used_by_name: str,
            user_node: BaseSchedulerNode | OutputNode,
            can_inplace: bool = False,
            is_weak: bool = False,
        ) -> None:
            # [FUNCTION DOC] HELPER: Registers user_node as a (possibly in-place or weak) user of the buffer, after rename resolution.
            name_to_users[rename(used_by_name)].append(
                NodeUser(user_node, can_inplace, is_weak)
            )

        # pyrefly: ignore [not-a-type, unsupported-operation]
        unbacked_symbol_to_origin_node: dict[sympy.Symbol, str | None] = {}

        # NB: None means that the dependency is on an input.  Don't actually
        # generate a dependency because if we do, Inductor will start trying
        # to free the unbacked int but that's pointless
        for val in V.graph.graph_inputs.values():
            if isinstance(val, sympy.Expr):
                for fs in val.free_symbols:
                    unbacked_symbol_to_origin_node[fs] = None
            elif isinstance(val, ir.TensorBox):
                # We also need to add symbols from input size as well because
                # AOTI doesn't lift the unbacked symints to inputs
                sym_size = [s for s in val.get_size() if isinstance(s, sympy.Expr)]
                for s in sym_size:
                    for fs in s.free_symbols:
                        unbacked_symbol_to_origin_node[fs] = None

        has_non_input_unbacked_defs = False
        for node in self.nodes:
            if node.node is None:
                raise AssertionError("expected node.node to be set")
            # unbacked symbols don't follow ordinary buffer dependencies, so
            # we track their def/uses separately
            unbacked_symbol_defs = sorted(
                node.node.get_unbacked_symbol_defs(), key=lambda x: x.name
            )
            for s in unbacked_symbol_defs:
                if not isinstance(s, sympy.Symbol):
                    raise AssertionError("expected s to be a sympy.Symbol")
                # Pick the first definer as canonical.  There may be multiple
                # because if a MultiOutputLayout buffer propagates an unbacked
                # symint to multiple outputs, they will all claim to def it.
                has_non_input_unbacked_defs = True
                if s not in unbacked_symbol_to_origin_node:
                    unbacked_symbol_to_origin_node[s] = node.get_name()

        for node in self.nodes:
            log.debug("scheduling %s", node.node)

            if has_non_input_unbacked_defs:
                if node.node is None:
                    raise AssertionError("expected node.node to be set")

                unbacked_symbol_uses = sorted(
                    node.node.get_free_symbol_uses(unbacked_only=True),
                    key=lambda x: x.name,
                )
                # if a kernel takes unbacked symints, register dependencies
                for s in unbacked_symbol_uses:
                    if s not in unbacked_symbol_to_origin_node:
                        raise AssertionError(
                            f"{s} not in {unbacked_symbol_to_origin_node}"
                        )
                    if (r := unbacked_symbol_to_origin_node[s]) is not None:
                        for buf in self.name_to_node[r].get_outputs():
                            node.add_fake_dep(StarDep(buf.get_name()))

            if (
                len(node.read_writes.writes) == 1
                and (dep := next(iter(node.read_writes.writes)))
                and isinstance(dep, MemoryDep)
            ):
                node_mode = dep.mode
            else:
                node_mode = None

            # Handle output mutations
            for buf in node.get_outputs():
                # a node will mutate either 0 or 1 buffers
                if len(buf.get_mutations()) > 1:
                    raise AssertionError(
                        f"expected at most one mutation, got {len(buf.get_mutations())}"
                    )
                for alt_name in buf.get_mutations():
                    alt_name = rename(alt_name)
                    is_ordering_only = buf.is_ordering_only()
                    if is_ordering_only:
                        add_user(alt_name, node, is_weak=True)
                        node.add_fake_dep(
                            WeakDep(alt_name, mutating_buf=buf.get_name(), is_fake=True)
                        )
                        continue
                    # this node must run after the prior writer
                    add_user(alt_name, node)
                    node.add_fake_dep(StarDep(alt_name, mode=node_mode))
                    for user in name_to_users[alt_name].items:
                        if user.get_name() == node.get_name():
                            continue

                        if not isinstance(user.node, BaseSchedulerNode):
                            raise AssertionError(
                                "expected user.node to be a BaseSchedulerNode"
                            )
                        for out_buf in user.node.get_outputs():
                            other_name = out_buf.get_name()
                            # this node must run after all prior readers
                            other_name = rename(other_name)
                            # Check if the prior reader is a true alias (view) vs a clone.
                            # Views share underlying storage with the mutated buffer, so we
                            # need a real dependency (is_fake=False) to keep the view's
                            # buffer alive until after this mutation completes. Clones have
                            # independent storage, so we only need an ordering dependency
                            # (is_fake=True) that won't extend their buffer lifetime.
                            is_alias = alt_name in out_buf.get_aliases()
                            node.add_fake_dep(
                                WeakDep(
                                    other_name,
                                    mutating_buf=buf.get_name(),
                                    is_fake=not is_alias,
                                )
                            )
                            add_user(other_name, node, is_weak=True)

            for add_dep in V.graph.additional_buffer_deps[node.get_name()]:
                add_user(add_dep, node, is_weak=True)
                # is_fake=True because these are control dependencies for ordering only,
                # they should not extend buffer lifetimes
                node.add_fake_dep(WeakDep(add_dep, node.get_name(), is_fake=True))

            for add_dep in V.graph.additional_star_deps[node.get_name()]:
                add_user(add_dep, node, is_weak=False)  # Strong dependency
                node.add_fake_dep(StarDep(add_dep))

            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                if not isinstance(read, WeakDep):
                    add_user(read.name, node, node.can_inplace(read))

            node.update_mutated_names(self.mutation_renames)

            # update our renaming scheme for the next iteration
            for buf in node.get_outputs():
                for alt_name in buf.get_mutations():
                    self.mutation_renames[rename(alt_name)] = buf.get_name()
                    self.mutation_renames[alt_name] = buf.get_name()
                    self.mutation_real_name[buf.get_name()] = (
                        self.mutation_real_name.get(alt_name, alt_name)
                    )

        # make sure outputs aren't dead-code-eliminated
        for buf_name in V.graph.get_output_names():
            log.debug("scheduling output %s", buf_name)
            add_user(buf_name, OutputNode(StarDep(buf_name)))

        # make sure unbacked symints aren't dead-code-eliminated
        if has_non_input_unbacked_defs:
            for out in V.graph.graph_outputs:
                for s in out.get_free_symbol_uses(unbacked_only=True):
                    if s not in unbacked_symbol_to_origin_node:
                        raise AssertionError(
                            f"{s} not in {unbacked_symbol_to_origin_node.keys()}"
                        )
                    if r := unbacked_symbol_to_origin_node[s]:
                        for buf_name in self.name_to_node[r].get_buffer_names():
                            log.debug(
                                "scheduling output %s for unbacked symint %s",
                                buf_name,
                                s,
                            )
                            add_user(buf_name, OutputNode(StarDep(buf_name)))

        # make sure input mutation isn't dead-code-eliminated
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(StarDep(name)))
                V.graph.mutated_inputs.add(name)
            elif name in V.graph.constants:
                # In AOTI, module parameters and buffers are not lifted as graph inputs
                add_user(name, OutputNode(StarDep(name)))

        inp_names = {
            name: index for index, name in enumerate(V.graph.graph_inputs.keys())
        }
        V.graph.mutated_input_idxs = [
            inp_names[name] for name in V.graph.mutated_inputs
        ]

        # copy users information onto the nodes
        for node in self.nodes:
            for buf in node.get_outputs():
                buf.set_users(name_to_users[buf.get_name()].items)

        for name in self.name_to_donated_buffer:
            self.name_to_donated_buffer[name].set_users(name_to_users[name].items)

        # For debug logging
        logbuf = IndentedBuffer()
        logbuf.splice("{")
        for key, value in name_to_users.items():
            with logbuf.indent():
                users = [v.get_name() for v in value.items]
                logbuf.splice(f"'{key}': {users},")
        logbuf.splice("}")
        str = logbuf.getrawvalue().rstrip()
        compute_dependencies_log.debug("BUFFER USER LIST\n")
        compute_dependencies_log.debug("===== AFTER SCHEDULING =====\n%s", str)

    def insert_memory_check_nodes(self) -> None:
        # [FUNCTION DOC] Debug pass enabled by test configs (track memory lifecycle): computes the memory timeline of the scheduled nodes, determines which
        # buffers are allocated and freed at each step, registers the memory-check op, and inserts after every node an extern memory-check node
        # that verifies the expected newly alive and dead buffers (final step flagged). Replaces self.nodes with the interleaved list.
        from .memory import (
            assign_memory_planning_info_for_scheduler_buffers,
            compute_memory_timeline,
            FreeableInputBuffer,
            get_freeable_input_buf,
        )

        graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
        name_to_freeable_input_buf: dict[str, FreeableInputBuffer] = (
            get_freeable_input_buf(self.nodes, graph_inputs)
        )

        if not torch._inductor.config.reorder_for_peak_memory:
            assign_memory_planning_info_for_scheduler_buffers(
                self.nodes, self.name_to_buf
            )

        graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
        buf_info_list, _, _ = compute_memory_timeline(
            self.nodes,
            name_to_freeable_input_buf,
            graph_outputs,
        )

        step_allocs_deallocs: list[tuple[list[str], list[str]]] = [
            ([], []) for _ in range(len(self.nodes))
        ]
        for buf_info in buf_info_list:
            # Skip zero-size buffers
            if buf_info.size_alloc == 0 and buf_info.size_free == 0:
                continue

            buf_name = buf_info.buffer.get_name()

            step_allocs_deallocs[buf_info.start_step][0].append(buf_name)
            step_allocs_deallocs[buf_info.end_step][1].append(buf_name)

        from torch._inductor.runtime.debug_utils import register_check_mem_op

        register_check_mem_op()

        def construct_mem_check_node(
            step_idx: int, is_final_step: bool
        ) -> ExternKernelSchedulerNode:
            # [FUNCTION DOC] HELPER: Builds an extern kernel scheduler node wrapping a memory-check kernel (cpu none-layout) with the expected newly alive and dead
            # buffer names and the final-step flag for the given step index; its operation name is derived from the node at that step.
            expected_newly_alive = step_allocs_deallocs[step_idx][0]
            expected_newly_dead = step_allocs_deallocs[step_idx][1]

            nontensor_args = [expected_newly_alive, expected_newly_dead, is_final_step]

            node = ir.MemoryCheckKernel(
                layout=NoneLayout(device=torch.device("cpu")),
                kernel=torch.ops._inductor_debug.check_memory_step.default,
                tensor_args=[],
                nontensor_args=nontensor_args,
                unflatten_args=lambda tensor_args, constant_args: (
                    tensor_args,
                    {
                        "alive": constant_args[0],
                        "dead": constant_args[1],
                        "is_final_step": constant_args[2],
                    },
                ),
            )
            node.operation_name = f"mem_check_{self.nodes[step_idx].get_name()}"
            return ExternKernelSchedulerNode(self, node)

        new_nodes = []

        for i, node in enumerate(self.nodes):
            new_nodes.append(node)
            new_nodes.append(
                construct_mem_check_node(i, is_final_step=(i == len(self.nodes) - 1))
            )

        self.nodes = new_nodes

    def dead_node_elimination(self) -> None:
        # [FUNCTION DOC] Removes nodes without real users (when config.use_dce is on). Walks nodes in reverse topological order: a buffer is dead if all its users are weak or
        # removed; a node with only dead buffers and no side effects is removed, marking its buffers and operation as removed in V.graph and dropping it from the
        # user lists of its inputs. Afterwards prunes weak dependencies on removed operations. Mutates self.nodes and V.graph.
        """
        Remove any nodes without users
        """
        if not config.use_dce:
            return

        # self.nodes is in topological order, so by iterating in reverse order
        # we have visited (and potentially removed) all users before visiting a
        # given node.
        updated_nodes = []
        for node in reversed(self.nodes):

            def can_eliminate_user(user: NodeUser) -> bool:
                # [FUNCTION DOC] HELPER: True if the user is weak or its node was already removed.
                return user.is_weak or user.get_name() in V.graph.removed_operations

            active_buffers = False
            for buf in node.get_outputs():
                can_eliminate = all(can_eliminate_user(u) for u in buf.users)
                if can_eliminate:
                    log.debug("removed dead buffer: %s", buf.get_name())
                    V.graph.removed_buffers.add(buf.get_name())
                else:
                    active_buffers = True

            can_eliminate = not node.has_side_effects() and not active_buffers

            if not can_eliminate:
                updated_nodes.append(node)
            else:
                # dead code
                log.debug("removed dead operation: %s", node.get_name())
                V.graph.removed_operations.add(node.get_name())
                for read in node.read_writes.reads:
                    if read.name in self.name_to_buf:
                        users = self.name_to_buf[read.name].users
                        self.name_to_buf[read.name].users = [
                            u for u in users if u.node.get_name() != node.get_name()
                        ]
        self.nodes = list(reversed(updated_nodes))

        # Prune any WeakDeps no longer needed
        for node in self.nodes:
            node.prune_weak_deps()

    def mode_requires_synchronization(self, mode: str | None) -> bool:
        # [FUNCTION DOC] Returns whether a store mode requires cross-thread synchronization (currently any non-None mode).
        """Check if store mode requires cross-thread synchronization."""
        return mode is not None  # Currently all non-None modes need sync

    def topological_sort_schedule(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Returns nodes in topological order by depth-first visiting unmet dependencies (sorted by name for determinism), considering only
        # dependencies produced within the given nodes.
        """
        Ensure nodes is in topologically sorted order
        """
        seen = OrderedSet[BaseSchedulerNode]()
        name_to_node: dict[str, BaseSchedulerNode] = dict()
        result: list[BaseSchedulerNode] = []

        def visit(n: BaseSchedulerNode) -> None:
            # [FUNCTION DOC] HELPER: Recursive depth-first visit appending a node after its in-set dependencies.
            if n not in seen:
                seen.add(n)
                for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                    # We only care about doing toposort within `nodes`
                    if dep.name not in name_to_node:
                        continue
                    visit(name_to_node[dep.name])
                result.append(n)

        for node in nodes:
            for name in node.get_buffer_names():
                name_to_node[name] = node
        for node in nodes:
            visit(node)
        return result

    def _enforce_switch_ordering(self) -> None:
        # [FUNCTION DOC] HELPER: Forces switch (conditional) nodes to keep their relative order by adding a fake weak dependency from each one on the buffer of the previous one,
        # avoiding collective ordering hangs.
        conditional_nodes = [n for n in self.nodes if isinstance(n.node, ir.Switch)]
        for i in range(1, len(conditional_nodes)):
            mutating_buf = next(iter(conditional_nodes[i].get_buffer_names()))
            prev_buf = next(iter(conditional_nodes[i - 1].get_buffer_names()))
            conditional_nodes[i].add_fake_dep(
                WeakDep(prev_buf, mutating_buf=mutating_buf, is_fake=True)
            )

    def _get_unmet_dep_nodes(self, snode: BaseSchedulerNode) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] HELPER: Returns the nodes (mapped to fused nodes) that define buffers the snode still depends on. Raises RuntimeError for unsupported node types.
        unmet_deps: OrderedSet[str] = OrderedSet()
        if isinstance(
            snode,
            (
                SchedulerNode,
                ExternKernelSchedulerNode,
                NopKernelSchedulerNode,
                FusedSchedulerNode,
                GroupedSchedulerNode,
            ),
        ):
            for dep in snode.unmet_dependencies:
                unmet_deps.add(dep.name)
        else:
            raise RuntimeError(
                f"get_unmet_dep_nodes is not implemented for {type(snode)}."
            )
        unmet_dep_ops = (self.name_to_buf[dep].defining_op_name() for dep in unmet_deps)
        return list(OrderedSet(self.name_to_fused_node[n] for n in unmet_dep_ops))

    def _topological_sort_nodes(self) -> list[list[BaseSchedulerNode]]:
        # [FUNCTION DOC] HELPER: Groups nodes by topological level (Kahn algorithm layers): each list holds nodes whose dependencies are all in earlier lists. Raises
        # AssertionError if a cycle prevents a complete sort.
        """
        Sort nodes by their topological order, return a list of node lists.
        """
        order = []
        nodes = dict.fromkeys(self.nodes, 0)
        children: dict[Any, Any] = {}
        for node in self.nodes:
            deps = self._get_unmet_dep_nodes(node)
            nodes[node] = len(deps)
            for dep in deps:
                c = children.get(dep, [])
                c.append(node)
                children[dep] = c

        zero_deg_nodes = [n for n, v in nodes.items() if v == 0]
        while zero_deg_nodes:
            order.append(zero_deg_nodes)
            for n in zero_deg_nodes:
                for user in children.get(n, []):
                    nodes[user] -= 1
                nodes.pop(n)
            zero_deg_nodes = [n for n, v in nodes.items() if v == 0]
        if nodes:
            raise AssertionError("Topological sort failed!")
        return order

    def compute_ancestors(self) -> None:
        # [FUNCTION DOC] Populates ancestors (set of defining operation names transitively depended on) for each node in topological order, skipping self-edges, and sets min_order and
        # max_order to the node's index.
        """
        Populate each node.ancestors
        """
        # note self.nodes is topologically sorted
        name_to_ancestors: dict[str, OrderedSet[str]] = {}
        for node in self.nodes:
            ancestors: OrderedSet[str] = OrderedSet()
            for dep in node.unmet_dependencies:
                dep_node_name = self.name_to_buf[dep.name].defining_op_name()
                # A node can transiently depend on a buffer it also writes (a
                # self-edge, e.g. from a mutating op whose read and write
                # resolve to the same buffer). A node is never its own
                # ancestor, so skip it rather than KeyError on itself.
                if dep_node_name == node.get_name():
                    continue
                ancestors.add(dep_node_name)
                ancestors |= name_to_ancestors[dep_node_name]
            name_to_ancestors[node.get_name()] = ancestors
            node.ancestors = ancestors

        for order, node in enumerate(self.nodes):
            node.min_order = order
            node.max_order = order

    def compute_input_distances(self) -> None:
        # [FUNCTION DOC] Sets min and max input distance of each node: 0 when it only depends on graph inputs or constants, else one plus the min/max of its dependencies' distances (self-edges ignored).
        """
        Populate each node's min/max_input_distance with the depth from graph
        inputs, measured as dependency hops before fusion. Nodes whose
        dependencies are all satisfied by graph inputs/constants have depth 0.
        """
        name_to_min_distance: dict[str, int] = {}
        name_to_max_distance: dict[str, int] = {}
        for node in self.nodes:
            # Skip self-edges (a node depending on a buffer it also writes); a
            # node is not at any distance from itself. See compute_ancestors.
            dep_ops = [
                op
                for dep in node.unmet_dependencies
                if (op := self.name_to_buf[dep.name].defining_op_name())
                != node.get_name()
            ]
            if not dep_ops:
                min_dist = 0
                max_dist = 0
            else:
                min_dist = min(name_to_min_distance[op] + 1 for op in dep_ops)
                max_dist = max(name_to_max_distance[op] + 1 for op in dep_ops)
            name_to_min_distance[node.get_name()] = min_dist
            name_to_max_distance[node.get_name()] = max_dist
            node.min_input_distance = min_dist
            node.max_input_distance = max_dist

    def merge_loops(self) -> None:
        # [FUNCTION DOC] When loop ordering after fusion is enabled, merges loops of every non-template SchedulerNode inside GPU (or halide-CPU) scheduler/fused nodes. Mutates the nodes' loop bodies.
        if not config.loop_ordering_after_fusion:
            return

        for node in self.nodes:
            # Even for CPU, if we are using the halide backend, we still need
            # the merge loops steps below
            if not isinstance(node, (SchedulerNode, FusedSchedulerNode)) or (
                not node.is_gpu() and config.cpu_backend != "halide"
            ):
                continue
            for snode in node.get_nodes():
                # merge loops for the scheduler node
                if not isinstance(snode, SchedulerNode) or snode.is_template():
                    continue

                snode.merge_loops()

                # Note that for CPU backend, merging loops will change
                # snode.group. It's fine for Triton backend.
                # But if we simplify update snode.group like this:
                #   group_fn = self.get_backend(snode.node.get_device()).group_fn
                #   snode.group = (snode.node.get_device(), group_fn(snode._sizes))
                # There is still an issue due to different snode in a
                # FusedSchedulerNode having different merged loops.
                # Skip CPU backend for now.

    def fuse_nodes(self, nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Repeats fusion rounds (at most 10, stopping when the node count stops shrinking or reaches one) with fuse_nodes_once, then, if loop ordering after
        # fusion or loop index inversion is enabled, performs one extra reorder round. Returns the new node list.
        """
        Combine eligible nodes into FusedSchedulerNodes.
        """
        with dynamo_timed(
            "Scheduler.fused_nodes", log_pt2_compile_event=True, log_waitcounter=True
        ):
            for i in range(10):
                old_len = len(nodes)
                fusion_log.debug(
                    "===== attempting fusion (%d/10): %d nodes =====",
                    i + 1,
                    old_len,
                )
                nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
                new_len = len(nodes)
                fusion_log.debug(
                    "completed fusion round (%d/10): fused %d nodes into %d nodes\n",
                    i + 1,
                    old_len,
                    new_len,
                )
                if new_len == old_len or new_len == 1:
                    fusion_log.debug(
                        "===== fusion complete (%d iterations) =====", i + 1
                    )
                    break

            if (
                config.loop_ordering_after_fusion
                or config.loop_index_inversion_in_fusion
            ):
                nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
            return nodes

    def process_grouped_nodes(self) -> None:
        # [FUNCTION DOC] Replaces every GroupedSchedulerNode in self.nodes by its unpacked (and internally fused) members.
        """
        Unpack GroupedSchedulerNode into regular nodes.
        """
        new_nodes: list[BaseSchedulerNode] = []
        for node in self.nodes:
            new_nodes.extend(
                node.unpack() if isinstance(node, GroupedSchedulerNode) else [node]
            )
        self.nodes = new_nodes

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        # [FUNCTION DOC] Benchmarks a list of fused nodes (must be non-empty) on the backend of the first node's device and returns (milliseconds, kernel path or source info).
        # Sets current_device as a side effect; timed under a compile-time autotune event.
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        if len(nodes) <= 0:
            raise AssertionError(f"expected nodes to be non-empty, got {len(nodes)}")
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        with dynamo_timed(
            "benchmark_fused_nodes",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="compile_time_autotune_time_us",
        ):
            return backend.benchmark_fused_nodes(nodes)

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[BaseSchedulerNode],
        benchmark_kernel: bool,
        hint_override: int | None = None,
    ) -> str:
        # [FUNCTION DOC] Generates kernel source code for a non-empty list of pre-fused nodes via the device backend (optionally benchmark kernel and hint override). Sets current_device.
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        if len(nodes) <= 0:
            raise AssertionError(f"expected nodes to be non-empty, got {len(nodes)}")
        device = nodes[0].get_device()
        self.current_device = device
        backend = self.get_backend(device)
        with dynamo_timed("generate_kernel_code_from_nodes"):
            return backend.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel, hint_override=hint_override
            )

    def benchmark_codegened_module(
        self, module: ModuleType, device: torch.device
    ) -> tuple[float, str]:
        # [FUNCTION DOC] Benchmarks an already compiled module on the given device through the backend and returns (milliseconds, path). Sets current_device.
        """
        Benchmark a compiled module and return the execution time
        in milliseconds on randomly generated inputs.
        """
        self.current_device = device
        backend = self.get_backend(device)
        with dynamo_timed("benchmark_codegened_module"):
            return backend.benchmark_codegened_module(module)

    def _has_layout_conflict_for_template(
        self, multi_node: ir.MultiTemplateBuffer
    ) -> bool:
        # [FUNCTION DOC] HELPER: Returns True if choosing a Triton template for multi_node conflicts with recorded buffer layout constraints (logs a warning). Flexible input layouts are frozen to
        # the expected strides as a side effect; an input already frozen to a different fixed layout is a conflict.
        """
        Check if selecting a Triton template would cause layout conflicts.
        Returns True if there's a conflict and we should fall back to ATen.
        """
        constraints = V.graph.buffer_layout_constraints
        if not constraints:
            return False

        log.debug("Node %s has constraints %s", multi_node, constraints)
        for inp in multi_node.inputs:
            # pyrefly: ignore [missing-attribute]
            inp_name = inp.get_name()

            if not torch._inductor.select_algorithm.should_use_layout_constraints(inp):
                continue

            # pyrefly: ignore [missing-attribute]
            layout = inp.layout
            expected_layout = constraints[inp_name]
            if isinstance(layout, ir.FlexibleLayout):
                # Freeze to the expected layout to avoid conflicts
                # pyrefly: ignore [missing-attribute]
                inp.freeze_layout_with_exact_strides(expected_layout.stride)
                # pyrefly: ignore [missing-attribute]
                layout = inp.layout

            if isinstance(layout, ir.FixedLayout) and expected_layout != layout:
                # Layout already frozen to a different layout - conflict
                log.warning(
                    "Layout conflict detected for %s: template expects %s but layout is frozen to %s",
                    inp_name,
                    expected_layout,
                    layout,
                )
                return True

        return False

    def finalize_multi_template_buffers(self) -> None:
        # [FUNCTION DOC] For every SchedulerNode wrapping a MultiTemplateBuffer that was not finalized by fusion, picks the best choice (the extern kernel choice when forced for tests or a
        # layout conflict exists, else the fastest). A Triton choice is finalized in place (per multi-kernel hint if configured); an extern choice replaces the node with a new
        # scheduler node via _replace_node. Forces benchmarking/compilation. Raises AssertionError if no extern choice exists for fallback.
        """
        Finalize a backing choice for MultiTemplateBuffers which did not already have a
        choice finalized through fusion. In the case of an extern choice, this will result
        in replacing the SchedulerNode.

        If a MultiTemplateBuffer did not have any fusion opportunities, finalizing a choice
        will force completion of compilation and benchmarking.
        """

        for i, node in enumerate(self.nodes):
            if isinstance(node, SchedulerNode) and isinstance(
                node.node, ir.MultiTemplateBuffer
            ):
                multi_node = node.node
                if (
                    not config.test_configs.force_extern_kernel_in_multi_template
                    and not self._has_layout_conflict_for_template(multi_node)
                ):
                    # TODO (PaulZhang12): Potentially support recompilation/benchmark for new strides
                    min_node_unfused, _ = multi_node.get_min_choice()
                else:
                    extern_choices = [
                        c
                        for c in multi_node.choices
                        if isinstance(
                            c,
                            torch._inductor.select_algorithm.ExternKernelCaller,
                        )
                    ]
                    if not extern_choices:
                        raise AssertionError("No extern kernel detected for fallback")
                    if len(extern_choices) > 1:
                        timings = multi_node.choice_timings()
                        min_node_unfused = min(extern_choices, key=lambda c: timings[c])
                    else:
                        min_node_unfused = extern_choices[0]

                if isinstance(
                    min_node_unfused,
                    torch._inductor.ir.TritonTemplateCallerBase,
                ):
                    # pyrefly: ignore [unbound-name]
                    if config.multi_kernel_hints:
                        callers: dict[int | None, TritonTemplateCallerBase] = {}
                        callers[None] = min_node_unfused

                        # pyrefly: ignore [unbound-name]
                        for hint in config.multi_kernel_hints:
                            timings = multi_node.choice_timings(hint_override=hint)
                            triton_timings = {
                                k: v
                                for k, v in timings.items()
                                if isinstance(k, TritonTemplateCallerBase)
                            }
                            choice = min(triton_timings.items(), key=lambda x: x[1])[0]
                            callers[hint] = choice

                        node.node.finalize_as_triton_callers(callers)
                    else:
                        node.node.finalize_as_triton_caller(min_node_unfused)
                    continue

                with ir.IRNode.current_origins(multi_node.origins):
                    out_tensorbox = min_node_unfused.output_node()
                out_storage = out_tensorbox.data  # type: ignore[union-attr]
                if not isinstance(out_storage, ir.StorageBox):
                    raise AssertionError("expected out_storage to be an ir.StorageBox")
                out_buffer = out_storage.data
                if not isinstance(out_buffer, ir.OperationBuffer):
                    raise AssertionError(
                        "expected out_buffer to be an ir.OperationBuffer"
                    )

                if multi_node.origin_node:
                    assign_origin_node(out_tensorbox, multi_node.origin_node)

                out_buffer.layout = multi_node.layout
                self._replace_node(out_buffer, multi_node, i, node)

    def _replace_node(
        self,
        out_buffer: ir.OperationBuffer,
        multi_node: ir.MultiTemplateBuffer,
        i: int,
        node: SchedulerNode,
    ) -> None:
        # [FUNCTION DOC] HELPER: Replaces the scheduler node i (wrapping a multi-template buffer) with a new scheduler node for out_buffer: swaps the IR buffers in the graph, updates
        # name maps, carries over mutation renames, users, order, ancestors, last usage and mempool assignments from the old node.
        _replace_operation_buffer(multi_node, out_buffer)
        new_scheduler_node = self.create_scheduler_node(out_buffer)

        self.nodes[i] = new_scheduler_node
        self.name_to_node[node.get_name()] = new_scheduler_node
        self.name_to_fused_node[node.get_name()] = new_scheduler_node

        # We need to reflect the mutation renames that were recorded in the original node
        mutation_renames = {}
        for dep in itertools.chain(node.read_writes.reads, node.unmet_dependencies):
            if real_name := self.mutation_real_name.get(dep.name, None):
                mutation_renames[real_name] = dep.name

        def rename_deps(deps: OrderedSet[Dep]) -> OrderedSet[Dep]:
            # [FUNCTION DOC] HELPER: Returns a set of dependencies renamed through the mutation renames.
            return OrderedSet(dep.rename(mutation_renames) for dep in deps)

        new_scheduler_node.unmet_dependencies = rename_deps(
            new_scheduler_node.unmet_dependencies
        )
        new_scheduler_node.read_writes.reads = rename_deps(
            new_scheduler_node.read_writes.reads
        )

        for new_out, old_out in zip(
            new_scheduler_node.get_outputs(), node.get_outputs()
        ):
            self.name_to_buf[old_out.get_name()] = new_out
            new_out.users = old_out.users

        new_scheduler_node.min_order = node.min_order
        new_scheduler_node.max_order = node.max_order
        new_scheduler_node.ancestors = node.ancestors
        new_scheduler_node.last_usage = node.last_usage

        mempool = self.node_to_mempool.get(node)
        if node not in self.node_to_mempool:
            mempool = self._get_node_mempool(node)
        # If replacement happens before _populate_mempool_assignments, the
        # later population pass must be able to recover the preserved context
        # from the replacement IR node itself.
        if new_scheduler_node.node is not None:
            typing.cast(ir.IRNode, new_scheduler_node.node).mempool = mempool
        self.node_to_mempool[new_scheduler_node] = mempool
        for buf in new_scheduler_node.get_buffer_names():
            self.buff_to_mempool[buf] = mempool
        if mempool is not None:
            self._mempool_nodes = True

    def _any_atomic_add(self, node_list: Sequence[BaseSchedulerNode]) -> bool:
        # [FUNCTION DOC] HELPER: True if any node has IR data with scatter mode atomic_add.
        return any(
            hasattr(n.node, "data")
            and n.node is not None
            and hasattr(n.node.data, "scatter_mode")
            and n.node.data.scatter_mode == "atomic_add"
            for n in node_list
        )

    def compile_kernel(
        self, nodes: Sequence[BaseSchedulerNode], hint_override: int | None = None
    ) -> tuple[LambdaFuture | None, ModuleType]:
        # [FUNCTION DOC] Generates benchmark kernel code for the nodes and loads it as a module. If the module has a triton kernel and a process pool is available, also starts an
        # asynchronous compile and returns its future. Returns (future or None, module).
        src_code = self.generate_kernel_code_from_nodes(
            nodes, benchmark_kernel=True, hint_override=hint_override
        )
        mod = PyCodeCache.load(src_code)

        if not hasattr(mod, "triton_"):
            return (None, mod)

        async_compile = torch._inductor.async_compile.AsyncCompile()
        if not async_compile.use_process_pool():
            fut = None
        else:
            fut = async_compile.triton(kernel_name="triton_", source_code=src_code)
            if not isinstance(fut, LambdaFuture):
                raise AssertionError("expected fut to be a LambdaFuture")

        return (fut, mod)

    def speedup_by_fusion(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusionResult:
        # [FUNCTION DOC] Decides whether fusing node1 and node2 is worthwhile and returns a FusionResult (immediate or deferred callable with a future).
        # Immediate results: fuse (no benchmark) when benchmark_fusion is off (unless a multi-template is involved), for non-triton templates, foreach
        # nodes, non-triton CPU, or atomic-add nodes outside multi-template; refuse atomic-add template epilogues when disabled by config, on layout
        # conflicts, for prologue fusion with atomic add, or when no choice compiles. For multi-template buffers: per multi-kernel hint, compile the fused
        # kernel for each triton choice, then (deferred) benchmark fused choices or use an estimated epilogue cost and register/occupancy heuristic, finalizing
        # the winning choice on the multi-template buffer (side effect) and returning whether it is profitable. Otherwise compiles the two unfused kernels and
        # the fused kernel in parallel and (deferred) benchmarks them, fusing if the fused time is lower; register spilling (infinite time) refuses,
        # and a loop-carried variable compile error allows fusion. May add a slow-fusion metric row.
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """

        is_multi_template = any(
            n.is_template()
            and isinstance(n.get_template_node(), ir.MultiTemplateBuffer)
            for n in (node1, node2)
        )
        atomic_add_template_epilogue = isinstance(
            node1.get_template_node(), ir.TritonTemplateBuffer
        ) and _is_atomic_add_mutation_epilogue(node2, check_config=False)
        if atomic_add_template_epilogue and not config.epilogue_fusion_with_atomic_add:
            return FusionResult.fuse(False)

        if not config.benchmark_fusion and not is_multi_template:
            return FusionResult.fuse(True)

        if (
            node1.is_template()
            and not isinstance(node1.get_template_node(), ir.TritonTemplateBuffer)
            or node1.is_foreach()
            or node2.is_foreach()
        ):
            return FusionResult.fuse(True)

        node_list_1 = node1.get_nodes()
        device = node_list_1[0].get_device()
        if not device:
            raise AssertionError("expected device to be set")

        # don't support benchmark fusion for CPU C++ backend right now.
        if device.type == "cpu" and config.cpu_backend != "triton":
            return FusionResult.fuse(True)

        node_list_2 = node2.get_nodes()
        node_list_fused = list(itertools.chain(node_list_1, node_list_2))
        has_atomic_add = self._any_atomic_add(node_list_fused)

        # We can not accurately benchmark kernel using atomic_add
        # due to how we generate random integer inputs.
        # Skip benchmarking them by allowing fusion. The
        # epilogue_fusion_with_atomic_add gate only applies to Triton template
        # epilogues, so keep the existing non-template behavior here.
        if has_atomic_add and not is_multi_template:
            return FusionResult.fuse(True)

        from triton.compiler.errors import CompilationError

        why = WhyNoFuse(node1, node2)

        device = node_list_fused[0].get_device()
        if device is None:
            raise AssertionError("expected device to be set")

        def log_fusion(ms_fused: float, ms1: float, ms2: float) -> None:
            # [FUNCTION DOC] HELPER: Logs at debug level whether fusing was assumed, a benchmarked speedup or a slowdown, given fused and unfused times in ms.
            if fusion_log.isEnabledFor(logging.DEBUG):
                if ms_fused == 0.0:
                    fusion_log.debug(
                        "can fuse (assumed): fusing %s with %s (benchmarking skipped)",
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                    )
                elif ms_fused < ms1 + ms2:
                    fusion_log.debug(
                        "can fuse (benchmark): fusing %s with %s cause %sx speedup",
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                        green_text(f"{(ms1 + ms2) / ms_fused:.3f}"),
                    )
                else:
                    fusion_log.debug(
                        "cannot fuse (benchmark): fusing %s with %s cause %sx slowdown",
                        node1.get_buffer_names(),
                        node2.get_buffer_names(),
                        red_text(f"{ms_fused / (ms1 + ms2):.3f}"),
                    )

        if is_multi_template and any(
            n.get_template_node() is not None for n in (node1, node2)
        ):
            epilogue_fusion = node1.get_template_node() is not None
            multi_node = (
                node1.get_template_node()
                if epilogue_fusion
                else node2.get_template_node()
            )
            if not isinstance(multi_node, ir.MultiTemplateBuffer):
                raise AssertionError(
                    "expected multi_node to be an ir.MultiTemplateBuffer"
                )
            # Check for layout conflicts before committing to Triton template
            if self._has_layout_conflict_for_template(multi_node):
                return FusionResult.fuse(False)

            hint_override_best_fusion_choice: dict[int | None, ir.ChoiceCaller] = {}
            if not has_atomic_add:
                for hint_override in config.multi_kernel_hints:
                    future_choices: list[
                        tuple[Any, LambdaFuture | None, ModuleType]
                    ] = []
                    choice_timings = multi_node.choice_timings(hint_override)
                    for choice, _ in sorted(choice_timings.items(), key=lambda x: x[1]):
                        if not isinstance(
                            choice,
                            torch._inductor.select_algorithm.TritonTemplateCaller,
                        ):
                            continue
                        with multi_node.swap_as_triton_caller(choice):
                            future_choices.append(
                                (
                                    choice,
                                    *self.compile_kernel(
                                        node_list_fused,
                                        hint_override=choice.hint_override,
                                    ),
                                )
                            )

                    min_ms_fused = float("inf")
                    ms_fused_choice: TritonTemplateCallerBase | None = None
                    new_timings = {}
                    for choice, future, mod_fused in future_choices:
                        try:
                            if future is not None:
                                future.result()
                        except Exception as e:
                            if fusion_log.isEnabledFor(logging.DEBUG):
                                fusion_log.debug(
                                    "Exception in compiling %s: %s",
                                    "prologue" if not epilogue_fusion else "epilogue",
                                    e,
                                )
                            continue
                        with multi_node.swap_as_triton_caller(choice):
                            ms_fused, path = self.benchmark_codegened_module(
                                mod_fused, device
                            )
                            new_timings[choice] = ms_fused
                            if ms_fused < min_ms_fused:
                                min_ms_fused = ms_fused
                                ms_fused_choice = choice
                    multi_node._choice_timings[hint_override] = new_timings
                    if ms_fused_choice is not None:
                        if not isinstance(ms_fused_choice, TritonTemplateCallerBase):
                            raise AssertionError(
                                f"expected TritonTemplateCallerBase, "
                                f"got {type(ms_fused_choice)}"
                            )
                        hint_override_best_fusion_choice[hint_override] = (
                            ms_fused_choice
                        )

            from torch._inductor.codegen.nv_universal_gemm import NVUniversalGemmCaller

            bench_epilogue = config.benchmark_epilogue_fusion
            num_fusible_callers = sum(
                isinstance(c, (TritonTemplateCallerBase, NVUniversalGemmCaller))
                for c in multi_node.choices
            )
            # Track if the choice timings can be retrieved async after compilation
            get_choice_timings_async = (
                use_pipelined_autotuning()
                and not bench_epilogue
                and num_fusible_callers <= config.max_epilogue_benchmarked_choices
            )

            ms1, ms2 = float("inf"), float("inf")
            min_choice: ir.ChoiceCaller | None = None
            if not get_choice_timings_async:
                choice_timings = multi_node.choice_timings()
                min_choice, ms1 = multi_node.get_min_choice()

                choice_timings_iter = sorted(
                    choice_timings.items(), key=operator.itemgetter(1)
                )
            else:
                # Use 0 for unfused time, won't be used as bench_epilogue
                # is guaranteed to be False here
                choice_timings_iter = [(c, 0) for c in multi_node.choices]

            from torch._inductor.codegen.simd import CantSplit

            def choice_supports_fusion(choice: ir.ChoiceCaller) -> bool:
                # [FUNCTION DOC] HELPER: True if the choice is a Triton template caller that can take part in the fusion (for prologue fusion its allowed prologue inputs must match the multi-template's).
                if not isinstance(
                    choice, torch._inductor.select_algorithm.TritonTemplateCaller
                ):
                    return False
                # For prologue fusion we check if the underlying template of the choice
                # supports all allowed prologue inputs. If not, we skip this choice in
                # the fusion benchmark.
                # TODO: Remove this check after all Triton templates support prologue fusion.
                # Currently, persistent+TMA Triton template does not due to the TMA-based loads.
                return not (
                    not epilogue_fusion
                    and hasattr(choice, "allowed_prologue_inps")
                    and choice.allowed_prologue_inps != multi_node.allowed_prologue_inps
                )

            def compile_without_benchmarking(
                choice: torch._inductor.select_algorithm.TritonTemplateCaller,
            ) -> bool:
                # [FUNCTION DOC] HELPER: Swaps the choice into the multi-template buffer and compiles the fused kernel (waiting for the async compile or precompiling). Returns False if the kernel cannot be split or
                # compilation fails (logged at debug level), else True.
                with multi_node.swap_as_triton_caller(choice):
                    try:
                        future, mod_fused = self.compile_kernel(
                            node_list_fused, hint_override=choice.hint_override
                        )
                        if future is not None:
                            future.result()
                        else:
                            mod_fused.triton_.precompile()
                    except CantSplit:
                        return False
                    except Exception as e:
                        if fusion_log.isEnabledFor(logging.DEBUG):
                            fusion_log.debug(
                                "Exception in compiling %s: %s",
                                "prologue" if not epilogue_fusion else "epilogue",
                                e,
                            )
                        return False
                return True

            if has_atomic_add:
                if not epilogue_fusion:
                    return FusionResult.fuse(False)

                for hint_override in [*config.multi_kernel_hints, None]:
                    choice_timings = multi_node.choice_timings(hint_override)
                    ms_fused_choice: TritonTemplateCallerBase | None = None
                    for choice, _ in sorted(
                        choice_timings.items(), key=operator.itemgetter(1)
                    ):
                        if not choice_supports_fusion(choice):
                            continue
                        triton_choice = typing.cast(
                            torch._inductor.select_algorithm.TritonTemplateCaller,
                            choice,
                        )
                        if compile_without_benchmarking(triton_choice):
                            ms_fused_choice = triton_choice
                            break
                    if ms_fused_choice is None:
                        return FusionResult.fuse(False)
                    hint_override_best_fusion_choice[hint_override] = ms_fused_choice

                if config.multi_kernel_hints:
                    multi_node.finalize_as_triton_callers(
                        hint_override_best_fusion_choice  # pyrefly: ignore [bad-argument-type]
                    )
                else:
                    best = hint_override_best_fusion_choice[None]
                    # pyrefly: ignore [bad-argument-type]
                    multi_node.finalize_as_triton_caller(best)
                return FusionResult.fuse(True)

            if bench_epilogue:
                ms2, path2 = (
                    self.benchmark_fused_nodes(node_list_2)
                    if epilogue_fusion
                    else self.benchmark_fused_nodes(node_list_1)
                )
            else:
                # By default, don't do prologue fusion. Generally slower
                if not epilogue_fusion:
                    return FusionResult.fuse(False)

                ms2 = node2._get_estimated_runtime()
                ms2_fused = _estimate_fused_epilogue_runtime(node1, node2, ms2)

            future_choices: list[tuple[Any, LambdaFuture | None, ModuleType]] = []
            template_choices = 0
            for choice, unfused_time in choice_timings_iter:
                is_triton = isinstance(
                    choice, torch._inductor.ir.TritonTemplateCallerBase
                )
                is_nvgemm = isinstance(choice, NVUniversalGemmCaller)

                if not is_triton and not is_nvgemm:
                    continue

                # pyrefly: ignore [missing-attribute]
                if is_nvgemm and not choice.supports_epilogue_fusion:
                    continue

                # NVGEMM doesn't support prologue fusion. Skip NVGEMM choices in
                # the prologue direction (epilogue_fusion is False when node1 is
                # the pointwise prologue, node2 is the template).
                if is_nvgemm and not epilogue_fusion:
                    continue

                # For prologue fusion we check if the underlying template of the choice
                # supports all allowed prologue inputs. If not, we skip this choice in
                # the fusion benchmark.
                # TODO: Remove this check after all Triton templates support prologue fusion.
                # Currently, persistent+TMA Triton template does not due to the TMA-based loads.
                if (
                    is_triton
                    and not epilogue_fusion
                    and hasattr(choice, "allowed_prologue_inps")
                    and choice.allowed_prologue_inps != multi_node.allowed_prologue_inps
                ):
                    continue

                if bench_epilogue and unfused_time >= ms1 + ms2:
                    break

                template_choices += 1
                if template_choices > config.max_epilogue_benchmarked_choices:
                    break

                try:
                    if is_triton:
                        # pyrefly: ignore [bad-argument-type]
                        with multi_node.swap_as_triton_caller(choice):
                            future_choices.append(
                                (
                                    choice,
                                    *self.compile_kernel(
                                        node_list_fused,
                                        hint_override=getattr(
                                            choice, "hint_override", None
                                        ),
                                    ),
                                )
                            )
                    elif is_nvgemm:
                        # pyrefly: ignore [missing-attribute]
                        with multi_node.swap_as_nvgemm_caller(choice):
                            future_choices.append(
                                (choice, *self.compile_kernel(node_list_fused))
                            )
                except CantSplit:
                    continue

            if len(future_choices) == 0:
                return FusionResult.fuse(False)

            def benchmark_when_ready() -> bool:
                # [FUNCTION DOC] HELPER: Deferred decision function for the multi-template fusion. Waits for compiled fused kernels (refreshing choice timings when obtained asynchronously), then either benchmarks each
                # fused choice (benchmark epilogue mode) or checks fusibility by unfused timing and a register/spill occupancy heuristic (NV universal GEMM choices accepted directly). If a fused choice is
                # profitable, finalizes the multi-template buffer on it (single triton caller, per-hint callers, or nvgemm) and returns True; otherwise False.
                nonlocal choice_timings, future_choices, ms1, min_choice, multi_node
                min_ms_fused = float("inf")
                ms_fused_choice = None
                new_timings = {}

                if get_choice_timings_async:
                    if not (
                        multi_node and isinstance(multi_node, ir.MultiTemplateBuffer)
                    ):
                        raise AssertionError(
                            "expected multi_node to be an ir.MultiTemplateBuffer"
                        )
                    choice_timings = multi_node.choice_timings()
                    min_choice, ms1 = multi_node.get_min_choice()

                    # Some choices can fail to benchmark, inf timing
                    future_choices = [
                        fut_choice
                        for fut_choice in future_choices
                        if fut_choice[0] in choice_timings
                    ]

                    future_choices = sorted(
                        future_choices,
                        key=lambda x: choice_timings[x[0]],
                    )
                # Benchmark each choice after compilation completes
                for choice, future, mod_fused in future_choices:
                    try:
                        if future is not None:
                            res = future.result()
                        elif not bench_epilogue:
                            if hasattr(mod_fused, "triton_"):
                                res = mod_fused.triton_
                                res.precompile()
                            else:
                                res = None
                        else:
                            res = None

                    # Ideally we would more narrowly catch Exceptions here but
                    # triton  will unpredictably error with valid prologue fusions
                    except Exception as e:
                        if fusion_log.isEnabledFor(logging.DEBUG):
                            fusion_log.debug(
                                "Exception in compiling %s: %s",
                                "prologue" if not epilogue_fusion else "epilogue",
                                e,
                            )
                        continue

                    if bench_epilogue:
                        is_nvgemm_choice = isinstance(choice, NVUniversalGemmCaller)
                        swap_ctx = (
                            # pyrefly: ignore [missing-attribute]
                            multi_node.swap_as_nvgemm_caller(choice)
                            if is_nvgemm_choice
                            # pyrefly: ignore [missing-attribute, bad-argument-type]
                            else multi_node.swap_as_triton_caller(choice)
                        )
                        with swap_ctx:
                            ms_fused, path = self.benchmark_codegened_module(
                                mod_fused,
                                # pyrefly: ignore [bad-argument-type]
                                device,
                            )
                            new_timings[choice] = ms_fused
                            if ms_fused < min_ms_fused:
                                min_ms_fused = ms_fused
                                ms_fused_choice = choice
                    else:
                        fusible_choice = (
                            min_choice == choice
                            or ms2 + ms1 > choice_timings[choice] + ms2_fused
                        )

                        is_nvgemm_choice = isinstance(choice, NVUniversalGemmCaller)
                        if is_nvgemm_choice and fusible_choice:
                            # NVGEMM register allocations are fixed by cutlass.operators;
                            # Triton's n_regs/n_spills heuristic doesn't apply.
                            ms_fused_choice = choice
                            break
                        elif res and fusible_choice:
                            choice.precompile()
                            # pyrefly: ignore [missing-attribute]
                            if not (res.launchers and choice.n_regs):
                                raise AssertionError(
                                    "expected res.launchers and choice.n_regs to be set"
                                )
                            # pyrefly: ignore [bad-index]
                            compiled_kernel = res.launchers[0]
                            # pyrefly: ignore [missing-attribute]
                            fused_n_regs = compiled_kernel.n_regs
                            # pyrefly: ignore [missing-attribute]
                            fused_n_spills = compiled_kernel.n_spills
                            should_fuse_epilogue = _fuse_epilogue(
                                ms1,
                                ms2,
                                choice.n_regs,
                                fused_n_regs,
                                fused_n_spills,
                                choice.bmreq.num_warps,
                                DeviceProperties.create(device),
                            )
                            if should_fuse_epilogue:
                                ms_fused_choice = choice
                                break

                if bench_epilogue:
                    log_fusion(min_ms_fused, ms1, ms2)

                if (
                    not bench_epilogue or min_ms_fused < (ms1 + ms2)
                ) and ms_fused_choice is not None:
                    is_nvgemm = isinstance(ms_fused_choice, NVUniversalGemmCaller)
                    if is_nvgemm:
                        # pyrefly: ignore [missing-attribute]
                        multi_node.finalize_as_nvgemm_caller(ms_fused_choice)
                    elif config.multi_kernel_hints:
                        hint_override_best_fusion_choice[None] = ms_fused_choice
                        # pyrefly: ignore [missing-attribute]
                        multi_node.finalize_as_triton_callers(
                            hint_override_best_fusion_choice
                        )
                    else:
                        # pyrefly: ignore [missing-attribute]
                        multi_node.finalize_as_triton_caller(ms_fused_choice)

                    if bench_epilogue:
                        # pyrefly: ignore [missing-attribute]
                        multi_node._choice_timings[None] = new_timings
                    return True
                else:
                    return False

            # Use a non-None future when available: handing None to from_callable
            # causes benchmark_when_ready to run synchronously, blocking any
            # remaining Triton async-compile overlap.
            deferred_future = next(
                (fut for _, fut, _ in future_choices if fut is not None), None
            )
            return FusionResult.from_callable(benchmark_when_ready, deferred_future)

        else:
            # Start parallel compilation for all three kernels
            future_and_mod_l1 = self.compile_kernel(node_list_1)
            future_and_mod_l2 = self.compile_kernel(node_list_2)
            future_and_mod_l1_fused = self.compile_kernel(node_list_fused)

            def benchmark_when_ready() -> bool:
                # [FUNCTION DOC] HELPER: Deferred decision function for plain nodes: waits for the three parallel compilations, benchmarks the unfused kernels and the fused kernel, records reasons through
                # WhyNoFuse for register spilling (infinite time), logs the result and (optionally) a slow-fusion metric row, and returns True if the fused kernel is faster than the sum of the two.
                # A missing triton config returns False; a loop-carried variable compile error returns True; other compile errors propagate.
                from torch._inductor.runtime.triton_heuristics import (
                    NoTritonConfigsError,
                )

                try:
                    # Wait for all compilations to complete
                    for fut in (
                        future_and_mod_l1[0],
                        future_and_mod_l2[0],
                        future_and_mod_l1_fused[0],
                    ):
                        if fut is not None:
                            fut.result()

                    ms1, path1 = self.benchmark_codegened_module(
                        future_and_mod_l1[1],
                        # pyrefly: ignore [bad-argument-type]
                        device,
                    )
                    if math.isinf(ms1):
                        why("register spilling of the first kernel")
                        return False

                    ms2, path2 = self.benchmark_codegened_module(
                        future_and_mod_l2[1],
                        # pyrefly: ignore [bad-argument-type]
                        device,
                    )
                    if math.isinf(ms2):
                        why("register spilling of the second kernel")
                        return False

                    ms_fused, path_fused = self.benchmark_codegened_module(
                        future_and_mod_l1_fused[1],
                        # pyrefly: ignore [bad-argument-type]
                        device,
                    )
                    if math.isinf(ms_fused):
                        why("register spilling of the fused kernel")
                        return False

                    log_fusion(ms_fused, ms1, ms2)

                    if (
                        is_metric_table_enabled("slow_fusion")
                        and ms_fused >= ms1 + ms2
                        and (path1, path2) not in self.logged_slow_fusion
                    ):
                        self.logged_slow_fusion.add((path1, path2))
                        get_metric_table("slow_fusion").add_row(
                            lambda: {
                                "kernel1_path": path1,
                                "kernel1_latency": ms1,
                                "kernel2_path": path2,
                                "kernel2_latency": ms2,
                                "fused_kernel_path": path_fused,
                                "fused_kernel_latency": ms_fused,
                                "slow_down_ratio": ms_fused / (ms1 + ms2),
                            }
                        )

                    return ms_fused < ms1 + ms2

                except NoTritonConfigsError:
                    return False

                except CompilationError as e:
                    if "Loop-carried variable" in str(e):
                        return True
                    raise

            return FusionResult.from_callable(
                callable_fn=benchmark_when_ready, future=future_and_mod_l1_fused[0]
            )

    def get_fused_node(self, node: BaseSchedulerNode) -> BaseSchedulerNode:
        # [FUNCTION DOC] Returns the (possibly fused) node currently containing the given node, looked up by its first name.
        "Look up the node in Scheduler name_to_fused_node"
        return self.name_to_fused_node[node.get_first_name()]

    def fuse_two_nodes(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        fused_nodes: OrderedSet[BaseSchedulerNode],
    ) -> BaseSchedulerNode:
        # [FUNCTION DOC] Fuses node1 and node2 (must be on the same device, else AssertionError) via the device backend, updates fused_nodes (removes both, adds the result), the name to fused node map, and stream and
        # mempool assignments of the new node. Returns the fused node.
        fusion_log.debug("fusing %s with %s", node1.get_name(), node2.get_name())

        device = node1.get_device()
        if node2.get_device() != device:
            raise AssertionError(
                f"expected node2 device to be {device}, got {node2.get_device()}"
            )
        node3 = self.get_backend(device).fuse(node1, node2)
        fused_nodes.remove(node1)
        fused_nodes.remove(node2)
        fused_nodes.add(node3)
        self.name_to_fused_node.update({n.get_name(): node3 for n in node3.get_nodes()})

        # Propagate stream assignment to the fused node so that subsequent
        # fusion rounds still respect stream boundaries.
        self.node_to_stream[node3] = self.get_node_stream(node1)
        self.node_to_mempool[node3] = self.node_to_mempool.get(node1)

        return node3

    def fuse_if_speedup(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        speedup_fn: Callable[[], bool],
        fused_nodes: OrderedSet[BaseSchedulerNode],
    ):
        # [FUNCTION DOC] Fuses node1 and node2 if the scheduler allows it, it would not create a cycle and speedup_fn returns true; returns whether fusion happened.
        if (
            self.can_fuse(node1, node2)
            and not self.will_fusion_create_cycle(node1, node2)
            and speedup_fn()
        ):
            self.fuse_two_nodes(node1, node2, fused_nodes)
            return True

        return False

    def _evaluate_pending_template_fusions(
        self,
        template_fusion_candidates: dict[BaseSchedulerNode, list[PendingFusion]],
        fused_nodes: OrderedSet[BaseSchedulerNode],
    ) -> None:
        # [FUNCTION DOC] Resolves deferred template fusions per candidate pointwise node. Repeatedly takes the next pending fusion of each candidate, skips ones whose template was already fused, performs
        # synchronous ones immediately and evaluates async ones as their compile futures complete, fusing if profitable (fuse_if_speedup); fused or exhausted candidates are removed from
        # template_fusion_candidates (mutated in place). Raises AssertionError on inconsistent epilogue/prologue pairs.
        """
        Evaluate pending template fusions for a set of fusion candidate nodes.
        The fusion candidate nodes are pointwise nodes as potential epilogue
        or prologue fusions
        """

        while template_fusion_candidates:
            template_futures: list[Future] = []
            future_to_pending_fusion: dict[
                Future, tuple[PendingFusion, BaseSchedulerNode]
            ] = {}
            fusions_to_remove: OrderedSet[BaseSchedulerNode] = OrderedSet()
            for candidate in template_fusion_candidates:
                if not (
                    candidate in template_fusion_candidates
                    and len(template_fusion_candidates[candidate]) >= 1
                ):
                    raise AssertionError(
                        "expected candidate to have at least one template fusion candidate"
                    )
                pending_fusion = template_fusion_candidates[candidate].pop(0)

                if len(template_fusion_candidates[candidate]) == 0:
                    fusions_to_remove.add(candidate)

                node1, node2 = pending_fusion.get_fusion_nodes()

                if node2 == candidate:
                    if not is_epilogue_fusion(node1, node2):
                        raise AssertionError(
                            "expected node1, node2 to be an epilogue fusion"
                        )
                    template_node = node1
                else:
                    if node1 != candidate:
                        raise AssertionError("expected node1 to equal candidate")
                    if not is_prologue_fusion(node1, node2):
                        raise AssertionError(
                            "expected node1, node2 to be a prologue fusion"
                        )
                    template_node = node2

                # template node fused with same class of pointwise (prologue/epilogue)
                # move onto next candidate as not fusible
                # TODO (PaulZhang12): Does not support fusions of templates with
                # multiple potential epilogues
                if self.get_fused_node(template_node) is not template_node:
                    continue

                if pending_fusion.future:
                    f = pending_fusion.future.future
                    if f is None:
                        raise AssertionError("expected f to be set")
                    template_futures.append(f)
                    future_to_pending_fusion[f] = (pending_fusion, candidate)
                else:
                    # Non AsyncCompile path, perform fusion
                    if self.fuse_if_speedup(
                        node1, node2, pending_fusion.callable_fn, fused_nodes
                    ):
                        fusions_to_remove.add(candidate)

            # Evaluate fusion candidates as async_compile completes
            for f in as_completed(template_futures):
                pending_fusion, cand = future_to_pending_fusion[f]
                if self.fuse_if_speedup(
                    self.get_fused_node(pending_fusion.node1),
                    self.get_fused_node(pending_fusion.node2),
                    pending_fusion.callable_fn,
                    fused_nodes,
                ):
                    fusions_to_remove.add(cand)

            for f in fusions_to_remove:
                template_fusion_candidates.pop(f)

    def _try_fusion_pairs(
        self,
        possible_fusion_pairs: list[tuple[BaseSchedulerNode, BaseSchedulerNode]],
        pending_fusions: dict[BaseSchedulerNode, PendingFusion],
        template_fusion_nodes: dict[BaseSchedulerNode, list[PendingFusion]],
        fused_nodes: OrderedSet[BaseSchedulerNode],
        is_reorder_round: bool,
    ):
        # [FUNCTION DOC] Tries to fuse each (node1, node2) candidate pair in profitability order. For each pair it first resolves pending fusions involving either node, maps both to their current fused nodes,
        # skips already seen template fusions, then, if can_fuse allows it and no cycle results, asks speedup_by_fusion. A deferred result is recorded: template fusions are queued per pointwise node in
        # template_fusion_nodes (and remembered in seen_template_fusions), others in pending_fusions for both nodes; an immediate yes fuses the nodes. Mutates the three containers and fused_nodes.
        def resolve_pending_fusions(
            node1: BaseSchedulerNode,
            node2: BaseSchedulerNode,
        ) -> None:
            # [FUNCTION DOC] HELPER: While either node's fused node has a pending fusion, pops it and evaluates its deferred decision; fuses the pending pair if profitable and no cycle would be created.
            # Raises AssertionError on inconsistent bookkeeping.
            # KNOWN ISSUE: the cycle check uses node1 and node2 (the pair being processed) rather than the pending pair node_key1 and node_key2 that is actually fused.
            while (
                self.get_fused_node(node1) in pending_fusions
                or self.get_fused_node(node2) in pending_fusions
            ):
                pending_fusion = pending_fusions.get(
                    self.get_fused_node(node1),
                    pending_fusions.get(self.get_fused_node(node2)),
                )
                if pending_fusion is None:
                    raise AssertionError("expected pending_fusion to be set")

                node_key1, node_key2 = pending_fusion.get_fusion_nodes()
                is_speedup = pending_fusion.callable_fn

                pending_fusions.pop(node_key1, None)
                pending_fusions.pop(node_key2, None)

                if self.get_fused_node(node_key1) is not node_key1:
                    raise AssertionError("expected node_key1 to be its own fused node")
                if self.get_fused_node(node_key2) is not node_key2:
                    raise AssertionError("expected node_key2 to be its own fused node")

                if not is_speedup() or self.will_fusion_create_cycle(node1, node2):
                    continue

                self.fuse_two_nodes(node_key1, node_key2, fused_nodes)

        for node1, node2 in possible_fusion_pairs:
            # if either node is in a pending fusion, resolve it.
            # since we iterate on potential fusions based on profitability
            # the first potential fusion should take precedence.
            resolve_pending_fusions(node1, node2)
            node1 = self.get_fused_node(node1)
            node2 = self.get_fused_node(node2)

            if (
                is_template_fusion(node1, node2)
                and (node1, node2) in self.seen_template_fusions
            ):
                continue

            if self.can_fuse(
                node1, node2, is_reorder_round
            ) and not self.will_fusion_create_cycle(node1, node2):
                fusion_res = self.speedup_by_fusion(node1, node2)
                if fusion_res.callable_fn is not None:
                    pending_fusion = PendingFusion(
                        callable_fn=fusion_res.callable_fn,
                        node1=node1,
                        node2=node2,
                        future=fusion_res.future,
                    )

                    if is_template_fusion(node1, node2):
                        if (node1, node2) in self.seen_template_fusions:
                            raise AssertionError(
                                "expected (node1, node2) to not already be a seen template fusion"
                            )
                        self.seen_template_fusions.add((node1, node2))

                        template_pw_node = template_fusion_pw_node(node1, node2)
                        if template_pw_node not in template_fusion_nodes:
                            template_fusion_nodes[template_pw_node] = []
                        template_fusion_nodes[template_pw_node].append(pending_fusion)
                    else:
                        pending_fusions[node1] = pending_fusion
                        pending_fusions[node2] = pending_fusion

                    continue

                if not fusion_res.should_fuse:
                    continue

                self.fuse_two_nodes(node1, node2, fused_nodes)

    def _finish_pending_fusions(
        self,
        fused_nodes: OrderedSet[BaseSchedulerNode],
        pending_fusions: dict[BaseSchedulerNode, PendingFusion],
    ):
        # [FUNCTION DOC] After the main fusion loop, resolves the remaining non-template pending fusions (each distinct speedup function once): fuses the pair if profitable via fuse_if_speedup. Raises
        # AssertionError if a pending node was already fused elsewhere.
        seen_pair_speedup_fn: OrderedSet[Callable[[], bool]] = OrderedSet()

        # Resolve pending fusions for non templates in case of benchmark_kernel=True
        for pending_fusion in pending_fusions.values():
            node_key1, node_key2 = pending_fusion.get_fusion_nodes()
            is_speedup_fn = pending_fusion.callable_fn

            if is_speedup_fn in seen_pair_speedup_fn or is_template_fusion(
                node_key1, node_key2
            ):
                continue

            seen_pair_speedup_fn.add(is_speedup_fn)

            if self.get_fused_node(node_key1) is not node_key1:
                raise AssertionError("expected node_key1 to be its own fused node")
            if self.get_fused_node(node_key2) is not node_key2:
                raise AssertionError("expected node_key2 to be its own fused node")

            self.fuse_if_speedup(node_key1, node_key2, is_speedup_fn, fused_nodes)

    def _handle_template_overlap(
        self,
        possible_fusions: list[tuple[BaseSchedulerNode, BaseSchedulerNode]],
        deferred_prologue_fusions: list[tuple[BaseSchedulerNode, BaseSchedulerNode]],
    ):
        # [FUNCTION DOC] HELPER: Splits out prologue fusions whose template node is also the template of an epilogue fusion candidate: those are appended to deferred_prologue_fusions (to be evaluated on the
        # template plus epilogue) and the remaining pairs are returned.
        # Potentially a prologue fusion might have the same template as an epilogue
        # the prologue fusion therefore has to be evaluated on the potential
        # fused template + epilogue
        epilogue_template_nodes = OrderedSet(
            [n1 for n1, n2 in possible_fusions if is_epilogue_fusion(n1, n2)]
        )
        new_possible_fusions = []
        for n1, n2 in possible_fusions:
            if is_prologue_fusion(n1, n2) and n2 in epilogue_template_nodes:
                deferred_prologue_fusions.append((n1, n2))
            else:
                new_possible_fusions.append((n1, n2))

        return new_possible_fusions

    def fuse_nodes_once(
        self,
        nodes: list[BaseSchedulerNode],
        is_reorder_round: bool,
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] One fusion round. Prunes redundant deps, gets candidate pairs (get_possible_fusions), defers overlapping prologue fusions when max-autotune is on, tries the pairs, finishes pending
        # non-template fusions, evaluates pending template fusions, then handles deferred prologue fusions. Returns the nodes sorted by original order and topologically sorted.
        """
        Combine eligible nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuse(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        self.prune_redundant_deps(nodes)
        fused_nodes = OrderedSet(nodes)
        if fusion_log.isEnabledFor(logging.DEBUG):
            fusion_log.debug("fuse_nodes_once, candidates:")
            for node in fused_nodes:
                fusion_log.debug("  %s", node.debug_str_short())

        # These are potential fusions which we are async compiling,
        # and which we will benchmark profitability of.
        # Maps node -> (is_speedup_fn, LambdaFuture, node1, node2)
        # Only used in the case of benchmark_kernel=True
        pending_fusions: dict[
            BaseSchedulerNode,
            PendingFusion,
        ] = {}

        template_fusion_nodes: dict[BaseSchedulerNode, list[PendingFusion]] = {}
        deferred_prologue_fusions: list[
            tuple[BaseSchedulerNode, BaseSchedulerNode]
        ] = []

        possible_fusions = self.get_possible_fusions(
            nodes,
            is_reorder_round,
        )

        if config.max_autotune_gemm or config.max_autotune:
            possible_fusions = self._handle_template_overlap(
                possible_fusions, deferred_prologue_fusions
            )

        self._try_fusion_pairs(
            possible_fusions,
            pending_fusions,
            template_fusion_nodes,
            fused_nodes,
            is_reorder_round,
        )
        self._finish_pending_fusions(fused_nodes, pending_fusions)

        self._evaluate_pending_template_fusions(template_fusion_nodes, fused_nodes)
        template_fusion_nodes.clear()

        if deferred_prologue_fusions:
            self._try_fusion_pairs(
                deferred_prologue_fusions,
                pending_fusions,
                template_fusion_nodes,
                fused_nodes,
                is_reorder_round,
            )
            self._evaluate_pending_template_fusions(template_fusion_nodes, fused_nodes)

        nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        nodes = self.topological_sort_schedule(nodes)
        return nodes

    @staticmethod
    def _distance_windows(
        nodes: list[BaseSchedulerNode],
        node_to_idx: dict[BaseSchedulerNode, int],
        max_distance: int,
    ) -> Iterator[list[BaseSchedulerNode]]:
        # [FUNCTION DOC] Static generator. Sorts nodes by baseline index and yields consecutive windows whose index span is at most max_distance; a negative max_distance yields everything as one window.
        """Sort `nodes` by baseline index, then yield groups whose span
        is at most `max_distance`. Start a new window whenever the next
        node would push the span past the limit. Negative `max_distance`
        means "no limit" — yield everything as one window.
        """
        ordered = sorted(nodes, key=lambda n: node_to_idx[n])
        if max_distance < 0:
            if ordered:
                yield ordered
            return
        window: list[BaseSchedulerNode] = []
        window_min = -1
        for n in ordered:
            idx = node_to_idx[n]
            if window and idx - window_min > max_distance:
                yield window
                window = []
            if not window:
                window_min = idx
            window.append(n)
        if window:
            yield window

    def create_combo_kernel_nodes(self, num_ck_nodes: int | None = None) -> None:
        # [FUNCTION DOC] Groups parallel nodes into combo (foreach) kernels. For each group from the grouping algorithm it filters combinable nodes, splits them into distance windows, and keeps windows where a combo
        # kernel is profitable (speedup_by_combo_kernel). With a memory threshold configured each window goes through the peak memory gate with halving retries; otherwise every window becomes a combo node.
        # Accepted combos replace their members in the node list (preserving stream and mempool), optionally limited to num_ck_nodes groups. Updates self.nodes, name_to_fused_node, stream and mempool maps, and logs.
        """Group parallel nodes into combo kernels.

        Each parallel group is split into windows whose baseline-index
        span is at most `combo_kernel_max_distance` (set the config to a
        negative value to disable splitting). If a peak-memory threshold is set,
        each window goes through the gate, which simulates the
        post-fusion peak and accepts the combo only if the peak stays
        under the threshold; rejected windows are halved and retried.
        Without a threshold, every window becomes a combo directly.
        """
        fused_nodes = OrderedSet(self.nodes)
        count = 0
        num_nodes_orig = len(self.nodes)
        log.debug("ComboKernels: Generating with num_ck_nodes = %s...", num_ck_nodes)
        enable_autotune = config.combo_kernels_autotune > 0

        abs_thr_gb = config.combo_kernel_peak_memory_increase_gb
        pct_thr = config.combo_kernel_peak_memory_pct_threshold
        memory_check = abs_thr_gb is not None or pct_thr is not None
        max_distance = config.combo_kernel_max_distance
        memory_sim_time = 0.0
        mem_ctx: ComboKernelMemoryContext | None = None
        if memory_check:
            mem_ctx = self._init_peak_memory_context()
            node_to_idx = mem_ctx.node_to_idx
        else:
            node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        output_order = node_to_idx.copy()

        def _register_accept(
            combo_node: ForeachKernelSchedulerNode,
            accepted: list[BaseSchedulerNode],
            num: int,
        ) -> None:
            # [FUNCTION DOC] HELPER: Records an accepted combo: increments the combo count, replaces the accepted nodes by the combo node in the fused set, orders it at the earliest member, updates the fused node map, and
            # copies the (single) stream and mempool of the members. Raises AssertionError if members span several streams or mempools.
            nonlocal count
            count += 1
            log.info(
                "ComboKernels: Combining %d nodes for %d-th group",
                len(accepted),
                num,
            )
            for node in accepted:
                fused_nodes.remove(node)
            fused_nodes.add(combo_node)
            output_order[combo_node] = min(output_order[node] for node in accepted)
            self.name_to_fused_node.update(
                {n.get_name(): combo_node for n in combo_node.get_nodes()}
            )
            accepted_streams = OrderedSet(self.get_node_stream(n) for n in accepted)
            if len(accepted_streams) != 1:
                raise AssertionError(
                    f"Combo kernel combines multiple streams: {list(accepted_streams)}"
                )
            self.node_to_stream[combo_node] = next(iter(accepted_streams))
            accepted_mempools = OrderedSet(
                self.node_to_mempool.get(n) for n in accepted
            )
            if len(accepted_mempools) != 1:
                raise AssertionError(
                    "Combo kernel combines multiple CUDA MemPool contexts: "
                    f"{list(accepted_mempools)}"
                )
            self.node_to_mempool[combo_node] = next(iter(accepted_mempools))

        for num, node_list in enumerate(
            ForeachKernelSchedulerNode.group_nodes_for_combo_kernels(self)
        ):
            if num_ck_nodes is not None and count > num_ck_nodes:
                break
            members = ForeachKernelSchedulerNode.combinable_nodes(node_list)
            if len(members) < 2:
                continue

            for window in Scheduler._distance_windows(
                members, node_to_idx, max_distance
            ):
                if num_ck_nodes is not None and count > num_ck_nodes:
                    break
                if len(window) < 2 or not self.speedup_by_combo_kernel(window):
                    continue
                if memory_check:
                    if mem_ctx is None:
                        raise AssertionError("expected mem_ctx to be set")
                    sim_start = time.perf_counter()
                    self._try_combo_with_halving(
                        window,
                        num,
                        mem_ctx,
                        enable_autotune=enable_autotune,
                        on_accept=_register_accept,
                    )
                    memory_sim_time += time.perf_counter() - sim_start
                else:
                    combo_node = ForeachKernelSchedulerNode(
                        window[0].scheduler,
                        window,
                        use_custom_partition_algo=True,
                        enable_autotune=enable_autotune,
                        per_subkernel_blocks=config.combo_kernel_per_subkernel_blocks,
                    )
                    _register_accept(combo_node, window, num)

        self.nodes = sorted(fused_nodes, key=output_order.__getitem__)
        self.nodes = self.topological_sort_schedule(self.nodes)
        log.info(
            "Generated ComboKernel nodes: %d ComboKernels, totally %d -> %d nodes",
            count,
            num_nodes_orig,
            len(self.nodes),
        )
        if memory_check:
            log.info(
                "ComboKernels memory-aware: %.3fs spent in peak simulation",
                memory_sim_time,
            )
        self.prune_redundant_deps(self.nodes)

    def _init_peak_memory_context(self) -> ComboKernelMemoryContext:
        # [FUNCTION DOC] HELPER: Builds the baseline ComboKernelMemoryContext: assigns memory planning info, computes the original memory timeline, peak, live memory before each step and node indices.
        """Build the immutable baseline state the gate compares against:
        original buffer lifetimes, original peak, and step indices.
        """
        from .memory import (
            assign_memory_planning_info_for_scheduler_buffers,
            assign_memory_planning_info_for_scheduler_nodes,
            compute_memory_timeline,
            get_freeable_input_buf,
            live_memory_before_steps_from_buf_info_list,
            peak_memory_from_buf_info_list,
        )

        graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
        graph_outputs = OrderedSet(V.graph.get_output_names())

        name_to_freeable = get_freeable_input_buf(self.nodes, graph_inputs)
        assign_memory_planning_info_for_scheduler_buffers(self.nodes, self.name_to_buf)
        assign_memory_planning_info_for_scheduler_nodes(
            self.nodes,
            self.name_to_fused_node,
            self.name_to_buf,
            name_to_freeable,
        )
        buf_info_list, _, _ = compute_memory_timeline(
            self.nodes, name_to_freeable, graph_outputs
        )
        baseline_peak, _ = peak_memory_from_buf_info_list(
            buf_info_list, len(self.nodes)
        )
        baseline_live_before = live_memory_before_steps_from_buf_info_list(
            buf_info_list, len(self.nodes)
        )

        return ComboKernelMemoryContext(
            graph_outputs=graph_outputs,
            node_to_idx={node: idx for idx, node in enumerate(self.nodes)},
            baseline_peak=baseline_peak,
            running_peak=baseline_peak,
            baseline_live_before=baseline_live_before,
        )

    def _try_combo_with_memory_check(
        self,
        group_nodes: list[BaseSchedulerNode],
        mem_ctx: ComboKernelMemoryContext,
        enable_autotune: bool,
    ) -> tuple[ForeachKernelSchedulerNode | None, int]:
        # [FUNCTION DOC] HELPER: Memory gate for combining group_nodes into one combo node. Simulates replacing the members (within the smallest index window containing them) by the combo at one step, estimates the
        # region peak memory, and accepts if the increase over the original baseline peak (the running peak is used for max) stays under the configured absolute (GB) and/or percent limits. Returns
        # (combo node, step) if accepted (and updates mem_ctx.running_peak) or (None, 0). Raises AssertionError if no limit is configured.
        """The gate: does fusing `group_nodes` into one combo keep peak
        memory under the threshold?

        Returns `(combo_node, combo_step)` if accepted, or `(None, 0)`
        if rejected. The running peak lives on `mem_ctx.running_peak`.

        The pretend rewrite only changes node order inside the window
        `[region_start, region_end]` (the smallest range containing all
        members). Inside that window:
          - the members collapse into one combo node, all at `combo_step`,
          - everything else is evaluated from the original schedule.
        Outside the window, nothing moves. Earlier accepts are accounted
        for only by `mem_ctx.running_peak`. The threshold is checked
        against `mem_ctx.baseline_peak` (the original graph peak) so total
        peak drift is capped.
        """
        from .memory import estimate_region_peak_memory

        combo_node = ForeachKernelSchedulerNode(
            group_nodes[0].scheduler,
            group_nodes,
            use_custom_partition_algo=True,
            enable_autotune=enable_autotune,
            per_subkernel_blocks=config.combo_kernel_per_subkernel_blocks,
        )
        # Wire the combo's pred_buffers from its members so the gate
        # simulator can read `node.mpi_node.pred_buffers` uniformly.
        combo_pred_buffers = OrderedSet()
        for m in group_nodes:
            combo_pred_buffers.update(m.mpi_node.pred_buffers)
        combo_node.mpi_node = MemoryPlanningInfoForNode(
            pred_buffers=combo_pred_buffers,
        )

        node_to_idx = mem_ctx.node_to_idx

        # Single pass over group_nodes: build the set and find the
        # min/max baseline index together.
        group_set: OrderedSet[BaseSchedulerNode] = OrderedSet()
        region_start = sys.maxsize
        region_end = -1
        for n in group_nodes:
            group_set.add(n)
            idx = node_to_idx[n]
            if idx < region_start:
                region_start = idx
            if idx > region_end:
                region_end = idx

        # Build the post-rewrite slice for [region_start, region_end].
        local_entries: list[_LocalEntry] = []
        inserted_combo = False

        def add_local(node: BaseSchedulerNode, cur: int, baseline: int) -> None:
            # [FUNCTION DOC] HELPER: Appends a local slice entry (current step, baseline index, node).
            local_entries.append(_LocalEntry(cur, baseline, node))

        for i in range(region_start, region_end + 1):
            n = self.nodes[i]
            if n in group_set:
                if not inserted_combo:
                    add_local(combo_node, region_start, i)
                    inserted_combo = True
                continue
            add_local(n, i, i)

        local_nodes = [
            e.node for e in sorted(local_entries, key=lambda e: (e.cur, e.baseline))
        ]
        local_nodes = self.topological_sort_schedule(local_nodes)

        new_step = {n: region_start + i for i, n in enumerate(local_nodes)}
        combo_step = new_step[combo_node]
        for n in group_nodes:
            new_step[n] = combo_step

        def step_of(node: BaseSchedulerNode) -> int:
            # [FUNCTION DOC] HELPER: Returns the simulated step of a node (rewritten step if moved, else its original index).
            if node in new_step:
                return new_step[node]
            return node_to_idx[node]

        # Combo windows are evaluated independently. The entry live set is
        # therefore the original-schedule live memory before this region starts.
        cur_memory = mem_ctx.baseline_live_before[region_start]

        region_peak = estimate_region_peak_memory(
            local_nodes,
            region_start=region_start,
            region_end=region_end,
            step_of=step_of,
            graph_outputs=mem_ctx.graph_outputs,
            cur_memory=cur_memory,
        )

        # Compare against the *original* baseline peak (not the running
        # peak) to cap total drift across many accepts.
        original_peak = mem_ctx.baseline_peak
        new_peak = max(mem_ctx.running_peak, region_peak)
        delta = new_peak - original_peak
        abs_thr_gb = config.combo_kernel_peak_memory_increase_gb
        pct_thr = config.combo_kernel_peak_memory_pct_threshold
        limits = [float(abs_thr_gb) * (1024**3)] if abs_thr_gb is not None else []
        if pct_thr is not None:
            limits.append(pct_thr * original_peak)
        if not limits:
            raise AssertionError("expected limits to be truthy")
        accept = delta <= min(limits)

        pct = (100.0 * delta / original_peak) if original_peak > 0 else 0.0
        if not accept:
            log.debug(
                "ComboKernels memory-aware: rejected %d nodes "
                "(peak delta %+d bytes = %.3f%%)",
                len(group_nodes),
                delta,
                pct,
            )
            return None, 0

        log.info(
            "ComboKernels memory-aware: accepted %d nodes "
            "(peak delta %+d bytes = %.3f%%)",
            len(group_nodes),
            delta,
            pct,
        )
        mem_ctx.running_peak = new_peak
        return combo_node, combo_step

    def _try_combo_with_halving(
        self,
        candidate: list[BaseSchedulerNode],
        num: int,
        mem_ctx: ComboKernelMemoryContext,
        *,
        enable_autotune: bool,
        on_accept: Callable[
            [ForeachKernelSchedulerNode, list[BaseSchedulerNode], int], None
        ],
    ) -> None:
        # [FUNCTION DOC] HELPER: Tries to combine a candidate window under the memory gate; on rejection splits it at the baseline-index midpoint and retries each half (when at least two nodes remain), calling
        # on_accept for each accepted combo. Uses an explicit stack, early half first.
        """Try the full candidate; on reject, halve at the
        baseline-index midpoint and try each half.
        """
        n2i = mem_ctx.node_to_idx
        # Push late then early so early pops first.
        stack: list[list[BaseSchedulerNode]] = [candidate]
        while stack:
            subset = stack.pop()
            if len(subset) < 2 or not self.speedup_by_combo_kernel(subset):
                continue

            combo_node, combo_step = Scheduler._try_combo_with_memory_check(
                self, subset, mem_ctx, enable_autotune
            )
            if combo_node is not None:
                on_accept(combo_node, subset, num)
                continue

            idxs = [n2i[n] for n in subset]
            lo, hi = min(idxs), max(idxs)
            if lo == hi:
                continue
            mid = (lo + hi) // 2
            early = [n for n in subset if n2i[n] <= mid]
            late = [n for n in subset if n2i[n] > mid]
            if 2 <= len(late) < len(subset):
                stack.append(late)
            if 2 <= len(early) < len(subset):
                stack.append(early)

    def prune_redundant_deps(self, nodes: list[BaseSchedulerNode]) -> None:
        # [FUNCTION DOC] Prunes redundant dependencies (see node method) of every node using the fused node map.
        for node in nodes:
            node.prune_redundant_deps(self.name_to_fused_node)

    def get_possible_fusions(
        self,
        nodes: list[BaseSchedulerNode],
        is_reorder_round: bool,
    ) -> list[tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        # [FUNCTION DOC] Finds all legal fusion pairs among nodes: pairs sharing a used buffer (and, with aggressive fusion, a group), limited to a configured number of following neighbors; foreach and template pairs are
        # also tried in reversed order. Applies highest priority filtering, sorts by fusion score descending and returns the list of (node1, node2).
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        possible_fusions = []
        seen = OrderedSet[tuple[BaseSchedulerNode, BaseSchedulerNode]]()

        def check_all_pairs(nodes: list[BaseSchedulerNode]) -> None:
            # [FUNCTION DOC] HELPER: Considers each node paired with a bounded window of following nodes, skipping seen pairs, and appends legal fusion pairs (reversing for template/foreach nodes when needed) to the result list.
            for node1_index, node1 in enumerate(nodes):
                for node2 in nodes[
                    node1_index + 1 : node1_index
                    + 1
                    + config.max_fusion_buffer_group_pairwise_attempts
                ]:
                    key = (node1, node2)
                    if key in seen:
                        continue
                    seen.add(key)

                    if self.can_fuse(node1, node2, is_reorder_round):
                        possible_fusions.append(key)
                    elif (node2.is_template() or node2.is_foreach()) and self.can_fuse(
                        node2, node1, is_reorder_round
                    ):
                        # foreach fusions and epilogue fusions are order dependent
                        possible_fusions.append((node2, node1))

        buffer_names_grouping = collections.defaultdict(list)
        for node in nodes:
            if self.unfusable_node(node):
                continue
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        if config.aggressive_fusion:
            group_grouping = collections.defaultdict(list)
            for node in nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        possible_fusions = self.get_possible_fusions_with_highest_priority(
            possible_fusions
        )
        possible_fusions.sort(key=self.score_fusion_key, reverse=True)
        fusion_log.debug("found %d possible fusions", len(possible_fusions))
        return possible_fusions

    def will_fusion_create_cycle(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Returns True (and logs the reason) if fusing node1 and node2 would create a cycle through other fused nodes: searches the ancestors of the pair, via fused nodes, for a path back to either node.
        """
        Finds whether there's a path from node1 to node2 (or vice-versa)
        caused indirectly by other fusions.
        """
        # since we are just returning boolean here, use slightly faster, unordered set
        visited = OrderedSet[FusedSchedulerNode]()

        def found_path(node: BaseSchedulerNode) -> bool:
            # [FUNCTION DOC] HELPER: Depth first search through fused nodes (each visited once) for a path to the pair, pruned when all of the fused node's operations are already among the pair's ancestors.
            # only fused nodes can introduce new ancestors.
            if isinstance(node, FusedSchedulerNode) and node not in visited:
                visited.add(node)
                if node.get_operation_names().issubset(combined_ancestors):
                    # All fusion outputs are in ancestors of node1 and node2, thus
                    # cannot introduce new path:
                    #
                    # 1. if output is neither descendant of node1 or node2, the
                    #        output cannot introduce a path
                    # 2. due to [can_fuse]: if WLOG output is descendant of node1, it cannot be
                    #        on path(node1->node2), hence it cannot be ancestor of node2
                    # 3. due to [acyclic]: if WLOG output is descendant of node1, it cannot be
                    #        ancestor of node1
                    return False
                else:
                    # continue DFS of new ancestors introduced by the fusion
                    return bool(combined_names & node.ancestors) or any(
                        found_path(self.name_to_fused_node[n])
                        for n in node.ancestors - combined_ancestors
                    )
            return False

        # as above - use slightly faster, unordered set
        combined_names = (
            node1.get_operation_names()._dict.keys()
            | node2.get_operation_names()._dict.keys()
        )
        combined_ancestors = (
            node1.ancestors._dict.keys() | node2.ancestors._dict.keys()
        ) - combined_names
        cycle = any(found_path(self.name_to_fused_node[n]) for n in combined_ancestors)
        if cycle:
            WhyNoFuse(node1, node2)("will create cycle")
        return cycle

    def can_fusion_increase_peak_memory(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Heuristic: True if fusing node1 and node2 can increase peak memory. Finds the buffers each node reads that have a single user (reusable if unfused), intersects them by buffer reuse key, and sums
        # the sizes of the common keys as the memory overhead (False if a size is not an integer). Returns True only if the overhead is statically greater than 32 times the bandwidth saving (score_fusion_memory).
        """
        Return true if fusing the two nodes can potentially increasing peak memory.

        The implementation is more like a heuristic since we don't really know if we are at peak
        or not when trying to fuse these two nodes. The order of nodes may change later which makes the
        peak memory estimation hard.

        Here is how we decide the LOWER BOUND of extra memory allocation if we fuse these 2 nodes:
        1. find all buffers read by each node with a single user. These buffers are supposed to
           be reused if we don't fuses these 2 nodes
        2. find the intersection of these buffers for the two node and sum the total buffer size.
           If we don't fuse these two nodes, we can at lease avoid this much memory allocation.
           Note that the extra memory allocation is not necessarily causing peak memory increase.
           This is just a heuristic.

        We return true only if the saving for fusion can not trade off the extra memory allocation.
        """

        from .codegen.wrapper import buffer_reuse_key

        def _find_single_user_inputs(
            node: BaseSchedulerNode,
        ) -> list[ir.Buffer]:
            # [FUNCTION DOC] HELPER: Returns the buffers read by the node that have exactly one user and a tensor output.
            output = []
            for rd in node.read_writes.reads:
                buf = self.name_to_buf.get(rd.name)
                if buf and len(buf.users) == 1 and buf.node.has_tensor_output():
                    output.append(buf.node)
            return output

        # Check inputs that can be potentially reused
        lhs_dep_nodes = _find_single_user_inputs(node1)
        rhs_dep_nodes = _find_single_user_inputs(node2)

        lhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in lhs_dep_nodes)
        rhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in rhs_dep_nodes)

        common_reuse_keys = lhs_reuse_keys.intersection(rhs_reuse_keys)

        memory_overhead = 0
        for key in common_reuse_keys:
            try:
                memory_overhead += int(key[2])
            except ValueError:
                # not an integer. Fallback is to fuse
                return False

        bw_saving = self.score_fusion_memory(node1, node2)

        # The factor 32 here is quite arbitrary.
        if V.graph.sizevars.statically_known_gt(memory_overhead, 32 * bw_saving):
            return True
        return False

    def fusion_prevent_too_many_reads_and_writes(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode, threshold: int
    ) -> bool:
        # [FUNCTION DOC] True if the fused node would have more than threshold unique I/O buffers. Counts the union of reads and writes of the two nodes, excluding node2 reads produced by node1 and node1 writes that
        # become internal (removable through fusion).
        # After fusion, we need to calculate the unique I/O buffers
        # accounting for buffers that become internal (removed through fusion)

        # Get all nodes that will be in the fused node
        fused_node_names = OrderedSet(
            [node.get_name() for node in node1.get_nodes()]
            + [node.get_name() for node in node2.get_nodes()]
        )

        # Calculate node2 reads that can be removed through fusion,
        # i.e. node2 reads that are outputs of node1
        node1_write_names = OrderedSet(dep.name for dep in node1.read_writes.writes)
        node2_read_names = OrderedSet(dep.name for dep in node2.read_writes.reads)
        reads_removed_through_fusion = node2_read_names & node1_write_names

        # Calculate node1 writes that can be removed through fusion,
        # i.e. node1 writes that are only read by node2
        writes_removed_through_fusion: OrderedSet[str] = OrderedSet()
        for write_dep in node1.read_writes.writes:
            if self.can_buffer_be_removed_through_fusion(
                write_dep.name, fused_node_names
            ):
                writes_removed_through_fusion.add(write_dep.name)

        # Get all unique reads (union of both nodes' reads)
        all_read_names = OrderedSet(
            dep.name for dep in node1.read_writes.reads
        ) | OrderedSet(dep.name for dep in node2.read_writes.reads)

        # Get all unique writes (union of both nodes' writes)
        all_write_names = OrderedSet(
            dep.name for dep in node1.read_writes.writes
        ) | OrderedSet(dep.name for dep in node2.read_writes.writes)

        # Remove reads that become internal
        unique_reads = all_read_names - reads_removed_through_fusion

        # Remove writes that become internal
        unique_writes = all_write_names - writes_removed_through_fusion

        # Get all unique buffer names (reads and writes combined, but no double counting)
        unique_io_buffers = unique_reads | unique_writes

        return len(unique_io_buffers) > threshold

    def are_long_distant_nodes(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Heuristic to avoid lengthening live ranges: True if the two nodes are more than 64 positions apart in the original order (max of cross distances between min/max order).
        """
        This function prevents fusion for nodes that can increase memory
        footprint. This problem is more common in horizontal fusion, where nodes
        that are far apart in the original order get fused, lengthening the live
        intervals of tensors. This is very evident in models with activation
        checkpointing, where the recomputed nodes from different checkpointed
        regions get fused and significantly increase the memory footprint.

        The current attempt is a quick, possibly hacky, heuristic to prevent the
        fusion of nodes that are far away in the original order.

        A better but difficult to implement heuristic would be to use live
        intervals of the buffers, find region of peak pressure in the original
        program and prevent fusion that crosses that peak region. We might need
        special care or good approximation in this implementation, as fusion of
        node changes live intervals, and re-computing live intervals and peak
        memory after each fusion can introduce large compilation overhead.
        """
        proximity_score = max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return proximity_score > 64

    def decide_fusion_fail_reason(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        common_buf_names: tuple[str, ...] | OrderedSet[str],
    ) -> str:
        # [FUNCTION DOC] Diagnostic: for each common buffer explains why the two nodes share no memory-compatible access (non-memory deps, different numel, broadcast, different offset, mismatched loop orders, or
        # unknown with layout) and returns the mapping of buffer name to reason as a string.
        """
        Try to decide reasons why fusion fail due to no shared memory even though
        there are common buffers.
        """
        reasons = {}
        node1_name2dep = {dep.name: dep for dep in node1.read_writes.reads_and_writes()}
        node2_name2dep = {dep.name: dep for dep in node2.read_writes.reads_and_writes()}

        for buf_name in common_buf_names:
            buf = V.graph.get_buffer(buf_name)
            lhs_dep = node1_name2dep[buf_name]
            rhs_dep = node2_name2dep[buf_name]

            if not isinstance(lhs_dep, MemoryDep) or not isinstance(rhs_dep, MemoryDep):
                reasons[buf_name] = (
                    f"not MemoryDep: {type(lhs_dep)} v.s. {type(rhs_dep)}"
                )
                continue

            if lhs_dep.get_numel() != rhs_dep.get_numel():
                reasons[buf_name] = (
                    f"different numel: {lhs_dep.get_numel()} v.s. {rhs_dep.get_numel()}"
                )
                continue

            # same numel but different MemoryDep.size. Should be broadcasting
            if sympy_product(lhs_dep.size) != sympy_product(rhs_dep.size):
                reasons[buf_name] = "broadcast"
                continue

            lhs_off = lhs_dep.get_offset()
            rhs_off = rhs_dep.get_offset()
            if lhs_off != rhs_off:
                # One example is in transformer, we use a concatenated linear layer
                # to project Q/K/V and then split the result. The 3 splits will
                # point to the same buffer with different offsets.
                reasons[buf_name] = f"different offset: {lhs_off} v.s. {rhs_off}"
                continue

            if (
                lhs_dep.normalize_with_stride_order()
                == rhs_dep.normalize_with_stride_order()
            ):
                reasons[buf_name] = f"Mismatch loop orders: {lhs_dep} v.s. {rhs_dep}"
                continue

            # Add more rules here
            layout_str = ""
            if not isinstance(buf, ir.TorchBindObject):
                layout_str = f"Layout: {buf.layout}"
            reasons[buf_name] = (
                f"Unknown reason: {lhs_dep} v.s. {rhs_dep}. {layout_str}"
            )

        return str(reasons)

    def _can_reindex_consumer_for_index_inversion(
        self,
        producer_write: MemoryDep,
        consumer_read: MemoryDep,
        consumer_write: MemoryDep,
        consumer: SchedulerNode,
        read_expr: sympy.Expr,
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if the consumer pointwise node can be re-indexed to a single flattened loop so index inversion can apply: not a reduction, same read/write size, matching flattened sizes with the producer
        # write, not already flat, contiguous write, a body with matching iteration variables, and an invertible flattened read.
        if consumer.is_reduction() or consumer_read.size != consumer_write.size:
            return False

        flat_size = sympy_product(producer_write.size)
        if not V.graph.sizevars.statically_known_equals(
            sympy_product(consumer_read.size), flat_size
        ) or not V.graph.sizevars.statically_known_equals(
            sympy_product(consumer._sizes[0]), flat_size
        ):
            return False
        if tuple(consumer._sizes[0]) == (flat_size,):
            return False

        body = consumer._body
        if body is None or not consumer_write.normalize().is_contiguous():
            return False

        iter_vars = body.vars[0]
        iter_sizes = body.sizes[0]
        if len(iter_vars) != len(iter_sizes):
            return False

        return (
            self._get_flattened_read_inverse(
                read_expr, tuple(iter_vars), tuple(iter_sizes), flat_size
            )
            is not None
        )

    def _get_flattened_read_inverse(
        self,
        read_expr: sympy.Expr,
        iter_vars: tuple[sympy.Symbol, ...],
        iter_sizes: tuple[sympy.Expr, ...],
        flat_size: sympy.Expr,
    ) -> tuple[sympy.Symbol, sympy.Expr] | None:
        # [FUNCTION DOC] HELPER: Substitutes the old iteration variables of the read expression by the decomposition of one flat variable and returns (flat variable, inverse expression) if the resulting flat read is invertible,
        # else None (also None for zero flat size).
        if V.graph.sizevars.statically_known_equals(flat_size, 0):
            return None

        # A flat reindex decomposes one new loop variable into the old loop domain.
        # Apply that substitution to the read without rebuilding the LoopBody.
        flat_var = _FLATTENED_READ_VAR
        flattened_read = sympy_subs(
            read_expr,
            dict(zip(iter_vars, decompose_index(flat_var, iter_sizes))),
        )

        inverse = self._get_indexing_inverse(flattened_read, flat_var, flat_size)
        if inverse is None:
            return None
        return flat_var, inverse

    def _get_indexing_inverse(
        self,
        read_expr: sympy.Expr,
        index_var: sympy.Symbol,
        index_size: sympy.Expr,
    ) -> sympy.Expr | None:
        # [FUNCTION DOC] HELPER: Returns the inverse of an indexing expression with respect to index_var over index_size (canonicalized to a shared symbol and simplified), or None if not invertible.
        # Canonicalize the loop variable so preflight and the rebuilt LoopBody
        # share one scheduler-local cache entry.
        canonical_read = sympy_subs(read_expr, {index_var: _FLATTENED_READ_VAR})
        canonical_read = V.graph.sizevars.simplify_with_ranges(
            sympy.expand(canonical_read), {_FLATTENED_READ_VAR: index_size}
        )
        inverse = self._get_canonical_indexing_inverse(canonical_read, index_size)
        if inverse is None:
            return None
        return sympy_subs(inverse, {_FLATTENED_READ_VAR: index_var})

    @cache_on_self_and_args("Scheduler")
    def _get_canonical_indexing_inverse(
        self, read_expr: sympy.Expr, index_size: sympy.Expr
    ) -> sympy.Expr | None:
        # [FUNCTION DOC] HELPER: Cached. Computes the inverse formula of a canonical read expression over the shared flat symbol via the inversion analysis; None if not invertible.
        from torch._inductor.invert_expr_analysis import generate_inverse_formula

        return generate_inverse_formula(read_expr, _FLATTENED_READ_VAR, index_size)

    def shared_data_after_inverting_indexing(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> int:
        # [FUNCTION DOC] Tries to make node2 fusable with node1 by inverting node2's indexing (node2 reads a producer write of node1 discontiguously and writes contiguously). Returns -1 when not applicable
        # (feature off, CPU, wrong node types, no shared buffer, more than one unmet dependency or read/write, non-memory deps, unsupported body, non-invertible read). May first reindex node2 to a flat loop and
        # retry. On success rewrites node2's read/write indexing expressions (mutating node2) and returns the fusion score from shared memory.
        # KNOWN ISSUE: the docstring says 0 is returned when the optimization is not applicable, but the code returns -1.
        """
        Attempts to enable fusion between two nodes by inverting indexing patterns.

        This optimization targets cases where node1 has a contiguous write and
        node2 has a contiguous write but discontiguous read. By inverting the
        indexing in node2's read and write operations, we can make them compatible
        with node1 for potential fusion.

        Args:
            node1: First scheduler node (source)
            node2: Second scheduler node (target for inversion)

        Returns:
            int: Fusion score if successful, 0 if optimization not applicable
        """

        if not config.loop_index_inversion_in_fusion:
            return -1

        if any(n.is_cpu() for n in [node1, node2]):
            return -1
        if not isinstance(node2, SchedulerNode):
            return -1
        if not isinstance(node2.node, ir.ComputedBuffer):
            return -1
        body = node2._body
        if body is None:
            return -1

        # Check for shared buffers between nodes
        node1_buffer_names = node1.read_writes.buffer_names()
        node2_buffer_names = node2.read_writes.buffer_names()
        common_buffer_names = node1_buffer_names & node2_buffer_names

        if not common_buffer_names:
            return -1

        # only invert if node1 is single unmet dep
        node2_unmet_dependencies = OrderedSet(
            dep.name for dep in node2.unmet_dependencies
        )
        if node2_unmet_dependencies - node1_buffer_names:
            return -1

        if len(node2_unmet_dependencies) > 1:
            return -1

        # Currently only handle single read/write operations
        if len(node2.read_writes.reads) != 1 or len(node2.read_writes.writes) != 1:
            return -1

        node2_read = next(iter(node2.read_writes.reads))
        node2_write = next(iter(node2.read_writes.writes))

        if not isinstance(node2_read, MemoryDep) or not isinstance(
            node2_write, MemoryDep
        ):
            return -1

        matching_node1_writes = [
            dep for dep in node1.read_writes.writes if dep.name == node2_read.name
        ]
        if len(matching_node1_writes) != 1:
            return -1

        node1_write = matching_node1_writes[0]

        if not isinstance(node1_write, MemoryDep):
            return -1

        # Index inversion supports one read and one write expression without subblocks.
        if len(body.indexing_exprs) != 2 or body.subblocks:
            return -1
        node2_read_exprs = OrderedSet(body.get_read_exprs())
        if len(node2_read_exprs) != 1:
            return -1
        read_expr = next(iter(node2_read_exprs))

        # Check the flattened read before rebuilding the consumer LoopBody.
        can_reindex = self._can_reindex_consumer_for_index_inversion(
            node1_write, node2_read, node2_write, node2, read_expr
        )
        if can_reindex:
            node2.apply_loop_reindexing([sympy_product(node1_write.size)])
            # Re-enter with refreshed dependencies; inverse lookup uses the cache.
            return self.shared_data_after_inverting_indexing(node1, node2)

        if not node2_write.is_contiguous():
            return -1

        # We are checking for compatibility with the normalized node1 write
        # then modifying node2 reads/writes. since the node1 write will be just used
        # for compatibility, while node2 will be used in actual modification, just
        # normalize node1 not node2.
        node1_write = node1_write.normalize()

        if (
            node1_write.index != node2_write.index
            and node1_write.size != node2_write.size
        ):
            return -1

        if node2_read.size != node2_write.size or len(node2_read.var_names) != 1:
            return -1

        if not ("index0" in body.indexing_exprs and "index1" in body.indexing_exprs):
            raise AssertionError("expected index0 and index1 in node2 indexing_exprs")

        # Determine which index is for reading vs writing
        if read_expr == body.indexing_exprs["index0"]:
            read_expr_index = "index0"
            write_expr_index = "index1"
        else:
            if read_expr != body.indexing_exprs["index1"]:
                raise AssertionError("expected read_expr to match node2 index1 expr")
            read_expr_index = "index1"
            write_expr_index = "index0"

        index_vars = body.vars[0]
        if len(index_vars) != 1:
            return -1

        inverse_formula = self._get_indexing_inverse(
            read_expr, index_vars[0], node2_read.size[0]
        )

        # formula is not invertible
        if inverse_formula is None:
            return -1

        # === Apply Inversion ===

        # Swap the indexing expressions using the inverse formula
        node2.apply_indexing_exprs(
            {
                read_expr_index: body.indexing_exprs[write_expr_index],
                write_expr_index: inverse_formula,
            }
        )

        # Calculate fusion score
        score = self.score_fusion_memory(node1, node2)
        if not isinstance(score, int):
            raise AssertionError("expected score to be an int")
        if score == 0:
            score = self._score_fusion_memory_by_fusable_read_write(node1, node2)

        fusion_log.info("Shared memory after inversion: %d", score)
        return score

    def shared_data_after_reordering_loop(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        # [FUNCTION DOC] Tries to reorder loops (and, if enabled, reindex a pointwise node to the reduction's iteration domain) so that node1 and node2 access common buffers compatibly. Returns the new shared memory score, or -1 if
        # nothing was recomputed (CPU, templates, no common buffers, nothing to reorder or reindexing disabled). Mutates loop state of the nodes.
        """
        Right now just greedily reorder the loop of node1 to be compatible with node2,
        but ideally we should have some heuristics to reorder the loop for node2
        to be compatible with node1 if that's more efficient.

        Return the amount of shared data re-computed in this method.
        If no such recomputation happens, return -1 (not return 0 since 0 is a valid
        amount of shared data).

        """

        # TODO Don't do loop reordering/reindexing for CPU for now.
        # Should debug more why it does not work for CPU codegen
        if any(n.is_cpu() for n in [node1, node2]):
            return -1

        # in some rare case, a template can be passed in.
        # Check test_interaction_with_multi_template in test_loop_ordering.py
        # and https://github.com/pytorch/pytorch/issues/165579
        if node1.is_template() or node2.is_template():
            return -1

        common_buffer_names = (
            node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
        )
        if not common_buffer_names:
            return -1

        if config.loop_ordering_after_fusion:
            score = self._try_reorder_loops_for_candidates(node1, node2)
            if score >= 0:
                return score

        # No reordering candidates found (or loop ordering disabled).
        # Try reindexing the pointwise to match the reduction's iteration
        # domain (e.g., [1024, 8192] -> [65536, 128] for RMS norm with
        # reshape), then retry loop reordering if enabled. The retry is
        # needed because FusedSchedulerNodes may have more loop vars than
        # the reindexed pointwise (e.g., 3 vs 2), and only the normalize()
        # comparison in _try_reorder_loops_for_candidates handles that
        # num_vars mismatch.
        if (
            not config.loop_reindexing_after_fusion
            or not self._try_reindex_pointwise_for_reduction(node1, node2)
        ):
            return -1

        if config.loop_ordering_after_fusion:
            score = self._try_reorder_loops_for_candidates(node1, node2)
            if score >= 0:
                return score

        return self.score_fusion_memory(node1, node2)

    def _try_reorder_loops_for_candidates(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> int:
        # [FUNCTION DOC] HELPER: Among common buffers whose accesses match after stride order normalization (preferring write-then-read pairs, then the largest buffer), reorders loops of the pointwise node (node1 first,
        # else node2) to align them. If the loop variable counts differ, succeeds only when the deps are equal after merging loops. Returns the fusion memory score if reordered, else -1 (also -1 when a write-read pair differs by
        # more than dimension order, there are no candidates, or both nodes are reductions).
        """
        Find common buffers with matching normalized stride order but different
        loop orders, and try to reorder loops to align them.
        """
        common_buffer_names = (
            node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
        )
        node1_reads = {dep.name: dep for dep in node1.read_writes.reads}
        node1_writes = {dep.name: dep for dep in node1.read_writes.writes}
        node2_reads = {dep.name: dep for dep in node2.read_writes.reads}
        node2_writes = {dep.name: dep for dep in node2.read_writes.writes}

        candidates = []
        for buffer_name in common_buffer_names:
            lhs_dep = node1_writes.get(buffer_name) or node1_reads[buffer_name]
            rhs_dep = node2_writes.get(buffer_name) or node2_reads[buffer_name]

            is_write_read = (
                buffer_name in node1_writes and buffer_name in node2_reads
            ) or (buffer_name in node2_writes and buffer_name in node1_reads)

            if (
                lhs_dep.normalize_with_stride_order()
                == rhs_dep.normalize_with_stride_order()
            ):
                candidates.append(
                    (
                        is_write_read,
                        V.graph.sizevars.optimization_hint(
                            lhs_dep.get_numel(), fallback=0
                        ),
                        lhs_dep,
                        rhs_dep,
                    )
                )
            elif is_write_read:
                # A write→read dep failed normalize_with_stride_order.
                # This could be a dimension order issue (reordering can
                # fix it) or a factorization issue (only reindexing can).
                # Distinguish by checking if the write dep's sizes are
                # a subset of the read dep's — if so, reordering the
                # read's loops could align them.
                w = node1_writes.get(buffer_name) or node2_writes.get(buffer_name)
                r = node2_reads.get(buffer_name) or node1_reads.get(buffer_name)
                if isinstance(w, MemoryDep) and isinstance(r, MemoryDep):
                    sv = V.graph.sizevars
                    w_sizes = w.normalize().size
                    r_sizes = r.normalize().size
                    if not all(
                        any(sv.statically_known_equals(ws, rs) for rs in r_sizes)
                        for ws in w_sizes
                    ):
                        return -1

        if len(candidates) == 0:
            return -1

        # Prefer write→read deps over shared reads. Among same
        # priority, pick the largest buffer.
        _is_wr, _numel, lhs_dep, rhs_dep = max(
            candidates, key=operator.itemgetter(0, 1)
        )

        if not isinstance(lhs_dep, MemoryDep) or not isinstance(rhs_dep, MemoryDep):
            return -1

        if lhs_dep.num_vars != rhs_dep.num_vars:
            # this can happen due to we don't merge loops.
            # We can not do loop reordering in this case right now
            # Simply returning true if the two Deps are the same after
            # normalization (merging loops)
            if lhs_dep.normalize() == rhs_dep.normalize():
                return self.dep_size_hint(lhs_dep)
            return -1

        reordered = False
        # Only reorder loops for pointwise for now
        if not node1.is_reduction():
            reordered = node1.reorder_loops_by_dep_pair(lhs_dep, rhs_dep)
        elif not node2.is_reduction():
            reordered = node2.reorder_loops_by_dep_pair(rhs_dep, lhs_dep)
        else:
            loop_ordering_log.debug(
                "Don't reorder loops since both nodes are reductions: %s v.s. %s",
                node1.get_name(),
                node2.get_name(),
            )

        return self.score_fusion_memory(node1, node2) if reordered else -1

    def _selected_tiling_memory(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> MemoryCoalescing | None:
        # [FUNCTION DOC] HELPER: Returns the memory access (coalescing) cost of the given nodes codegen'd as one kernel using the selected tiling, or None when unknown (analysis disabled or inapplicable, foreach, non-scheduler
        # or CPU nodes, or no analysis). Results are cached per tuple of nodes and their loop state generation.
        """Memory accessed by these nodes codegened as one kernel, or None.

        None means the coalescing cost is unknown - either the analysis does not
        apply, or the tiling was forced or fell back and so was never scored.
        """
        from .codegen.simd import SIMDScheduling
        from .tiling_utils import analyze_memory_coalescing_for_nodes

        if (
            not nodes
            or not config.triton.coalesce_tiling_analysis
            or config.triton.prefer_nd_tiling
            or any(node.is_foreach() for node in nodes)
        ):
            return None

        snodes = [subnode for node in nodes for subnode in node.get_nodes()]
        if not snodes or not all(isinstance(node, SchedulerNode) for node in snodes):
            return None

        if any(node.is_cpu() for node in snodes):
            return None

        # The fusion search asks this for the same nodes over and over while
        # pairing them up (on one model, 1160 calls over 53 distinct nodes).
        # The answer is a function of the nodes' loop state, which
        # _loop_state_gen identifies, so key on that. snodes are leaf
        # SchedulerNodes (checked above), which the scheduler keeps for its
        # lifetime, so keying on the nodes themselves retains nothing extra.
        # pyrefly: ignore[missing-attribute]
        cache_key = tuple((sn, sn._loop_state_gen) for sn in snodes)
        cached = self._tiling_memory_cache.get(cache_key, _TILING_MEMORY_MISS)
        if cached is not _TILING_MEMORY_MISS:
            return cached

        analysis = analyze_memory_coalescing_for_nodes(snodes)
        if analysis is None:
            self._tiling_memory_cache[cache_key] = None
            return None

        reduction = max(snodes, key=lambda node: int(node.is_reduction()))
        _, (numel, rnumel) = reduction.group
        result = SIMDScheduling.select_tiling_with_memory(
            snodes, numel, rnumel, analysis
        ).memory
        self._tiling_memory_cache[cache_key] = result
        return result

    def _reindexing_regresses_memory_coalescing(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        unfused_memory: tuple[MemoryCoalescing, ...] | None,
    ) -> bool:
        # [FUNCTION DOC] HELPER: True if the fused kernel after reindexing has worse estimated memory cost than the two unfused kernels. Costs are weighted coalescing costs; when DRAM bandwidth is valid they are converted to time
        # (with a launch overhead for the unfused pair), otherwise compared directly. Returns False if the unfused memory is unknown or the fused memory cannot be computed. Logs rejections.
        """Did reindexing make the fused kernel's memory access worse?

        `unfused_memory` is measured before reindexing, so this compares the two
        original kernels against the single reindexed one.
        """
        if unfused_memory is None:
            return False
        fused_memory = self._selected_tiling_memory([node1, node2])
        if fused_memory is None:
            return False

        # The fused analysis omits removable intermediates, so saved
        # materialization traffic offsets coalescing regressions here.
        unfused_cost = sum(memory.weighted_cost() for memory in unfused_memory)
        fused_cost = fused_memory.weighted_cost()
        try:
            dram_gbps = get_gpu_dram_gbps()
            valid_dram_gbps = math.isfinite(dram_gbps) and dram_gbps > 0
        except Exception:
            dram_gbps = 0
            valid_dram_gbps = False
        if valid_dram_gbps:
            # GB/s is numerically bytes/ns, so division converts the
            # byte-equivalent costs to estimated time in nanoseconds.
            unfused_time_ns = (
                unfused_cost / dram_gbps + _REINDEXING_FUSION_LAUNCH_OVERHEAD_NS
            )
            fused_time_ns = fused_cost / dram_gbps
            regresses = fused_time_ns > unfused_time_ns
        else:
            regresses = fused_cost > unfused_cost
        if regresses:
            loop_ordering_log.debug(
                "rejecting reindex of %s and %s: unfused memory %s (cost=%d), "
                "fused memory %s (cost=%d)",
                node1.get_name(),
                node2.get_name(),
                unfused_memory,
                unfused_cost,
                fused_memory,
                fused_cost,
            )
            return True
        return False

    def _try_reindex_pointwise_for_reduction(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Reindexes the loops of a pointwise node to the (numel, rnumel) iteration groups of a reduction node so shared reads get identical index expressions. Returns True if reindexing was applied and kept. Not applied
        # for CPU, foreach, two reductions/no reductions, non-scheduler sub-nodes, mismatched total numel, incompatible ranges or when already in that form. Mutates the pointwise node's loops, then rolls back (returns False) if no common
        # dependency matches afterwards or if the memory coalescing estimate regresses; refreshes dependencies as needed.
        """
        Reindex a pointwise's iteration loops to match a reduction's
        groups. After reindexing, the shared reads have identical index
        expressions, enabling the codegen to CSE loads.

        Returns True if reindexing was applied.
        """
        from .codegen.simd import SIMDKernel

        # Keep this consistent with shared_data_after_reordering_loop(): CPU
        # reindexing is not validated yet.
        if node1.is_cpu() or node2.is_cpu():
            return False

        if node1.is_foreach() or node2.is_foreach():
            return False

        if node1.is_reduction() and not node2.is_reduction():
            reduction_node, pw_node = node1, node2
        elif node2.is_reduction() and not node1.is_reduction():
            reduction_node, pw_node = node2, node1
        else:
            return False

        _, groups = reduction_node.group
        red_numel = typing.cast(sympy.Expr, groups[0])
        red_rnumel = typing.cast(sympy.Expr, groups[1])
        target_numel = red_numel * red_rnumel

        if not all(isinstance(sn, SchedulerNode) for sn in pw_node.get_nodes()):
            return False
        snodes = typing.cast(list[SchedulerNode], pw_node.get_nodes())

        # All snodes must have the same total iteration numel matching
        # the reduction's numel * rnumel so they can be reindexed identically.
        if not all(
            V.graph.sizevars.statically_known_equals(
                sympy_product(sn._sizes[0]), target_numel
            )
            for sn in snodes
        ):
            return False

        if not all(
            SIMDKernel.is_compatible((red_numel, red_rnumel), sn.get_ranges())
            for sn in snodes
        ):
            return False

        # Nothing to reindex if the pointwise already uses the reduction split.
        target_iter_sizes = (red_numel, red_rnumel)
        if all(tuple(sn._sizes[0]) == target_iter_sizes for sn in snodes):
            return False

        # Measure each node's memory access before mutating any loops, so the
        # coalescing guard below can compare against the un-reindexed kernels.
        unfused_memory: tuple[MemoryCoalescing, ...] | None = None
        memory = tuple(self._selected_tiling_memory([node]) for node in (node1, node2))
        if all(m is not None for m in memory):
            unfused_memory = typing.cast("tuple[MemoryCoalescing, ...]", memory)

        # Local rollback is still needed even with _LoopMutationTracker: this
        # helper is also used by shared_data_after_reordering_loop(), where a
        # failed reindex attempt returns -1 and the caller may keep evaluating
        # fusion within the same can_fuse() call.
        rollback_snapshot = _LoopStateSnapshot.create((pw_node,))

        for sn in snodes:
            sn.apply_loop_reindexing([red_numel, red_rnumel])

        if isinstance(pw_node, FusedSchedulerNode):
            pw_node.group = snodes[0].group
            refresh_group_node_dependencies(pw_node)

        # Verify reindexing actually increases shared deps.
        common_names = (
            node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
        )
        n1_deps = {dep.name: dep for dep in node1.read_writes.reads_and_writes()}
        n2_deps = {dep.name: dep for dep in node2.read_writes.reads_and_writes()}
        has_benefit = any(
            self.deps_match_normalized(n1_deps[name], n2_deps[name])
            for name in common_names
        )
        if not has_benefit:
            rollback_snapshot.restore()
            return False

        # TODO: Measure fused memory first so coalesced fusions skip per-node analysis.
        # Reindexing the pointwise onto the reduction's split can make its own
        # accesses uncoalesced. Undo the reindex when that costs more than the
        # traffic saved by fusing.
        if self._reindexing_regresses_memory_coalescing(node1, node2, unfused_memory):
            rollback_snapshot.restore()
            return False

        # When loop ordering is disabled, re-extract deps with
        # normalize=True so variable names are canonical. This is
        # safe because no further loop reordering will occur.
        # Without this, reindexed deps use different var names
        # (e.g. c0 vs d0) causing exact dep comparisons to fail.
        if not config.loop_ordering_after_fusion:
            for sn in snodes:
                sn.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)
            if isinstance(pw_node, FusedSchedulerNode):
                refresh_group_node_dependencies(pw_node)

        return True

    def unfusable_node(self, node: BaseSchedulerNode) -> bool:
        # [FUNCTION DOC] True if the node can never be fused: no-op nodes that are neither templates nor multi-output template outputs, and extern nodes except templates, multi-output template outputs and user Triton kernels that allow
        # epilogue fusion.
        """
        Is this node unfusable under any conditions.
        """
        if isinstance(node, NopKernelSchedulerNode):
            return not node.is_template() and not is_output_of_multi_outputs_template(
                node.node
            )
        if isinstance(node, ExternKernelSchedulerNode):
            if isinstance(node.node, ir.UserDefinedTritonKernel):
                return not node.node.can_fuse_epilogue()
            return not node.is_template() and not is_output_of_multi_outputs_template(
                node.node
            )
        return False

    def check_prologue_fusion_heuristics_fusable(
        self,
        prologue_node: BaseSchedulerNode,
        template_node: BaseSchedulerNode,
        why: WhyNoFuse,
    ) -> bool:
        # [FUNCTION DOC] Heuristics to skip predictably slow prologue fusions. Always allowed for quantization ops invoked by the user. Rejects (recording the reason in why) when the prologue reads more than 1.1 times the bytes it writes,
        # when it only pads (unaligned reads), or when it needs fp32 upcasts for a low precision template. Returns whether fusion should be tried.
        """
        Heuristics to avoid benchmarking predictably slow prologue fusions
        """
        # user opt into more aggressive prologue fusion, don't use heuristics
        if prologue_node.get_operation_names() <= V.graph.invoke_quant_ops:
            return True

        read_bytes = prologue_node.get_read_buffer_sizes()
        write_bytes = prologue_node.get_write_buffer_sizes()

        # Initially, only do fusions which will result in fewer memory accesses inside of the template to avoid
        # potential bad cache behavior and shared memory use.
        # we also want to avoid benchmarking reliably unprofitable fusions like downcasts from fp32 -> fp16 inside kernel.
        # allowing gathers by allowing increasing write_bytes by small factor
        # TODO - make configurable per input, for instance, bias can fuse fp32 -> fp16 profitably

        BYTES_THRESHOLD_MULTIPLIER = 1.1
        if read_bytes > (write_bytes * BYTES_THRESHOLD_MULTIPLIER):
            why("prologue fusion will not increase amount of bytes read in kernel")
            return False

        # we want to avoid attempting to fuse predictably unprofitable prologues
        # such as increasing the unaligned reads or writes.
        # TODO - would be nice to generalize this, however, we would need more explicit
        # knowledge of memory access patterns in the TritonTemplate in order to know
        # the stride order to check alignment.
        origins = tuple(
            e.target
            for n in prologue_node.get_nodes()
            if n.node is not None
            for e in n.node.get_origins()
            if e.op == "call_function"
        )
        if origins == (torch.ops.aten.constant_pad_nd.default,):
            why(
                "prologue fusion will not increase attempt to fuse in padding bc it increases unaligned reads"
            )
            return False

        def low_prec_fp(dtype: torch.dtype) -> bool:
            # [FUNCTION DOC] HELPER: True if the dtype is a floating point type of at most two bytes.
            return dtype.itemsize <= 2 and dtype.is_floating_point

        template_buf = template_node.get_template_node_or_throw()
        if (
            not template_buf.is_multi_outputs_template()
            and low_prec_fp(template_buf.dtype)
            and not prologue_node.can_codegen_in_low_precision()
        ):
            why(
                "prologue fusion that must be upcast to fp32 not profitable for low precision templates"
            )
            return False

        return True

    def get_expand_dim_for_pointwise_nodes(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> tuple[int, SchedulerNode, sympy.Expr] | None:
        # [FUNCTION DOC] Decides whether two small pointwise nodes that differ in exactly one iteration dimension can be fused by expanding the smaller node along that dimension. Requires plain scheduler nodes over computed buffers
        # without aliasing, mutation, reductions or reusable inputs, equal reduction sizes, one write each below the small memory threshold, and a statically known smaller size. Returns (dimension, node to expand, new size) or None.
        """
        Fusing two small pointwise nodes significantly reduces kernel overhead
        and launch overhead. However, slightly different sizes would prevent fusion.
        Here, we decide if expanding sizes of one node is profitible by allowing
        fusion, and returns the dimension to expand, node with smaller sizes,
        and new size after expand.
        """
        # only support scheduler node
        if not isinstance(node1, SchedulerNode) or not isinstance(node2, SchedulerNode):
            return None

        # only support computued buffer
        if not (
            isinstance(node1.node, ir.ComputedBuffer)
            and isinstance(node2.node, ir.ComputedBuffer)
        ):
            return None

        # does not support mutation yet since relying on index mod to handle
        # out-of-boundary access.
        if node1.has_aliasing_or_mutation() or node2.has_aliasing_or_mutation():
            return None

        # skip halide which does not support mod for index
        if config.cpu_backend == "halide":
            return None

        # only support pointwise nodes with the same reduction size
        n1_sizes, n2_sizes = node1._sizes, node2._sizes
        n1_iter_sizes, n1_reduce_sizes = n1_sizes
        n2_iter_sizes, n2_reduce_sizes = n2_sizes
        if (
            node1.is_reduction()
            or node2.is_reduction()
            or n1_reduce_sizes != n2_reduce_sizes
            or len(n1_iter_sizes) != len(n2_iter_sizes)
        ):
            return None

        # only support nodes with 1 write for simplification
        if len(node1.read_writes.writes) > 1 or len(node2.read_writes.writes) > 1:
            return None

        # When memory access is small, reducing gpu kernel overhead is profitable over
        # slightly larger memory access.
        node1_write_memory = self.dep_size_hint(next(iter(node1.read_writes.writes)))
        node2_write_memory = self.dep_size_hint(next(iter(node2.read_writes.writes)))
        if (
            max(node1_write_memory, node2_write_memory)
            > config.small_memory_access_threshold
        ):
            return None

        # does not support reinplace since `index % boundary` may lead to
        # race condition
        def has_reusable_buffer(node: BaseSchedulerNode) -> bool:
            # [FUNCTION DOC] HELPER: True if any input of the node could be reused in place by it (so expansion could cause races).
            for read in node.read_writes.reads:
                input_buf: SchedulerBuffer | SchedulerDonatedBuffer | None
                if read.name in self.name_to_donated_buffer:
                    input_buf = self.name_to_donated_buffer[read.name]
                else:
                    input_buf = self.name_to_buf.get(read.name)

                if (
                    input_buf
                    and V.graph.wrapper_code.can_reuse(input_buf, node)
                    and not isinstance(input_buf.defining_op, NopKernelSchedulerNode)
                ):
                    return True
            return False

        if has_reusable_buffer(node1) or has_reusable_buffer(node2):
            return None

        # only support nodes with 1 mismatch dimension
        mismatch_dimensions = []
        for idx, (n1_size, n2_size) in enumerate(zip(n1_iter_sizes, n2_iter_sizes)):
            if n1_size != n2_size:
                mismatch_dimensions.append(idx)

        if len(mismatch_dimensions) != 1:
            return None

        mismatch_dim = mismatch_dimensions[0]
        mismatch_size1, mismatch_size2 = (
            n1_iter_sizes[mismatch_dim],
            n2_iter_sizes[mismatch_dim],
        )
        if V.graph.sizevars.statically_known_lt(mismatch_size1, mismatch_size2):
            return mismatch_dim, node1, mismatch_size2
        elif V.graph.sizevars.statically_known_lt(mismatch_size2, mismatch_size1):
            return mismatch_dim, node2, mismatch_size1
        else:
            return None

    def _producer_output_names_read_by_consumer(
        self, producer: BaseSchedulerNode, consumer: BaseSchedulerNode
    ) -> OrderedSet[str]:
        # [FUNCTION DOC] HELPER: Returns the (mutation-renamed) names of buffers produced by producer that consumer reads.
        producer_buf_names = OrderedSet(
            self.mutation_renames.get(name, name)
            for name in producer.get_buffer_names()
        )
        return OrderedSet(
            name
            for dep in consumer.read_writes.reads
            if (name := self.mutation_renames.get(dep.name, dep.name))
            in producer_buf_names
        )

    def _nested_index_equivalent_dep_names(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> OrderedSet[str] | None:
        # [FUNCTION DOC] HELPER: For a dependent reduction pair that can fuse as nested reductions, returns the names of producer outputs read by the consumer (to be matched by equivalent index); otherwise None.
        if not NestedReduction.is_candidate(node1, node2):
            return None
        # These names feed both the score bridge and relaxed vertical dep
        # matching. The score bridge exists only so V.choices.can_fuse does not
        # reject legal equivalent-index deps before vertical legality runs, so it
        # must use the same fully legal nested relation as can_fuse_vertical.
        # TODO: split cheap score prefiltering from legality if this shows up in
        # compile-time profiles, without letting score-only names relax legality.
        if not NestedReduction.can_fuse(node1, node2):
            return None
        return self._producer_output_names_read_by_consumer(node1, node2)

    def _can_fuse_nested_reduction_append(
        self,
        grouped_node: BaseSchedulerNode,
        other: BaseSchedulerNode,
        pointwise_domains: Sequence[
            tuple[BaseSchedulerNode, NestedReduction.PointwiseDomain]
        ],
        *,
        can_reorder: bool,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Checks whether other can be appended to the grouped stage of a nested reduction: collects the grouped-node buffers read by parent-full domain consumers and runs the normal fusion check with relaxed
        # index matching for exactly those buffers.
        index_equivalent_dep_names: OrderedSet[str] = OrderedSet()
        grouped_buf_names = OrderedSet(
            self.mutation_renames.get(name, name)
            for name in grouped_node.get_buffer_names()
        )
        for sn, domain in pointwise_domains:
            if domain is not NestedReduction.PointwiseDomain.PARENT_FULL:
                continue
            reads = (
                self.mutation_renames.get(dep.name, dep.name)
                for dep in sn.read_writes.reads
            )
            index_equivalent_dep_names.update(
                dep_name for dep_name in reads if dep_name in grouped_buf_names
            )
        # Parent-full consumers may read grouped-stage outputs through
        # broadcasted parent-tile views. Relax matching only for those named
        # producer outputs; all other deps use normal vertical legality.
        return self._can_fuse(
            grouped_node,
            other,
            can_reorder=can_reorder,
            index_equivalent_dep_names=index_equivalent_dep_names,
        )

    def can_fuse(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        can_reorder: bool = False,
        allow_mix_order_reduction: bool = True,
    ) -> bool:
        # [FUNCTION DOC] Public fusion legality check for node1 then node2 (optionally in reorder round, optionally allowing mix order reduction). Wraps the check in a loop mutation tracker so speculative loop reorderings/reindexings
        # are rolled back when the result is False. Returns bool.
        """Determine if node1 and node2 can be combined into a single fused node.

        Speculative loop mutations (reordering, reindexing) are automatically
        rolled back if the fusion decision ultimately fails.
        """
        tracker = _LoopMutationTracker.create((node1, node2))
        can_fuse = self._can_fuse_impl(
            node1,
            node2,
            can_reorder=can_reorder,
            allow_mix_order_reduction=allow_mix_order_reduction,
        )
        tracker.finish(rollback=not can_fuse)
        return can_fuse

    def _can_fuse_impl(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        can_reorder: bool = False,
        allow_mix_order_reduction: bool = True,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Thin wrapper delegating to the internal fusion legality check.
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """
        return self._can_fuse(
            node1,
            node2,
            can_reorder=can_reorder,
            allow_mix_order_reduction=allow_mix_order_reduction,
        )

    def _can_fuse(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        can_reorder: bool = False,
        allow_mix_order_reduction: bool = True,
        index_equivalent_dep_names: OrderedSet[str] | None = None,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Core fusion legality check for node1 followed by node2. Returns False for the same node, different streams or mempools, or when node2 is a nested/mix-order fused node; delegates to
        # FusedNestedReductions / FusedMixOrderReductions can_fuse_with when node1 is one. Rejects (recording reasons through WhyNoFuse): strict-sum ordering violations, grouped/nop/extern nodes that are not eligible,
        # user Triton kernel epilogue conditions (unary in-place pointwise, same layout, no free-symbol uses beyond loads, no other users of the mutated buffer), data dependencies that put node2 before node1, prologue
        # fusion conditions for templates (enabled, pointwise producer, allowed inputs, functional, single-use chain, heuristics), template epilogue conditions (aliasing/mutation, reduction epilogue, config), explicitly
        # disabled buffers and device mismatch. Then computes a shared-memory fusion score (optionally expanding a small pointwise dimension, reordering/reindexing loops, or inverting indexing, which may mutate
        # nodes) and defers to the choices heuristics plus the backend: vertical fusion if node2 depends on node1 (with a reindexing retry), otherwise horizontal fusion.
        if node1 is node2:
            return False
        if index_equivalent_dep_names is not None:
            index_equivalent_dep_names = OrderedSet(
                self.mutation_renames.get(name, name)
                for name in index_equivalent_dep_names
            )

        # Prevent fusion across stream boundaries
        if self._has_multi_stream_nodes():
            stream1 = self.get_node_stream(node1)
            stream2 = self.get_node_stream(node2)
            if stream1 != stream2:
                return False
        if self._has_mempool_nodes():
            mempool1 = self.node_to_mempool.get(node1)
            mempool2 = self.node_to_mempool.get(node2)
            if mempool1 != mempool2:
                return False

        if isinstance(node1, FusedNestedReductions):
            return node1.can_fuse_with(node2, can_reorder=can_reorder)
        if isinstance(node2, FusedNestedReductions):
            return False

        if isinstance(node1, FusedMixOrderReductions):
            return node1.can_fuse_with(node2)
        if isinstance(node2, FusedMixOrderReductions):
            # We don't fuse something before a FusedMixOrderReductions
            # right now
            return False

        why = WhyNoFuse(node1, node2)

        if node1.is_template() and node2.has_strict_sum():
            why("template fusion does not preserve strict sum ordering")
            return False

        if (
            (node1.has_strict_sum() or node2.has_strict_sum())
            and node1.is_reduction()
            and node2.is_reduction()
        ):
            why("reduction fusion does not preserve strict sum ordering")
            return False

        if node1.is_template() and self.get_backend(
            node1.get_device()
        ).can_fuse_multi_outputs_template(node1, node2):
            return True
        if node1.is_template() and self.get_backend(
            node1.get_device()
        ).can_fuse_reduction_epilogue(node1, node2):
            return True

        if isinstance(node1, GroupedSchedulerNode) or isinstance(
            node2, GroupedSchedulerNode
        ):
            why("grouped node must not be fused with other nodes")
            return False
        if isinstance(node1, NopKernelSchedulerNode) and not node1.is_template():
            why("node1 is nop")
            return False

        if isinstance(node1, ExternKernelSchedulerNode):
            if not isinstance(node1.node, ir.UserDefinedTritonKernel):
                why("node1 is extern but not a triton kernel")
                return False

            if not node1.node.can_fuse_epilogue():
                why("node1's triton kernel doesn't support epilogue fusion")
                return False

            if not isinstance(node2, SchedulerNode):
                why("node1 is extern but node2 is not SchedulerNode")
                return False
            if not isinstance(node2.node, ComputedBuffer):
                why("node1 is extern but node2.node is not SchedulerNode")
                return False
            if not isinstance(node2.node.data, Pointwise):
                why("node1 is extern but node2.node.data is not Pointwise")
                return False

            if len(node1.node.mutation_outputs) != 1:
                raise AssertionError(
                    f"expected one mutation output, got {len(node1.node.mutation_outputs)}"
                )
            written_buffer_name = node1.node.mutation_outputs[0].name

            # The epilogue must be an in-place, unary pointwise operation.
            # Any other tensor/s would require additional load expressions.
            epilogue_reads = list(node2.read_writes.reads)
            epilogue_writes = list(node2.read_writes.writes)
            if (
                len(epilogue_reads) != 1
                or len(epilogue_writes) != 1
                or epilogue_reads[0].name != written_buffer_name
            ):
                why("epilogue is not a unary read of the output buffer")
                return False

            write_dep = epilogue_writes[0]
            read_dep = epilogue_reads[0]
            if not isinstance(read_dep, MemoryDep):
                raise AssertionError(f"expected MemoryDep, got {type(read_dep)}")
            if not isinstance(write_dep, MemoryDep):
                raise AssertionError(f"expected MemoryDep, got {type(write_dep)}")
            if read_dep.index != write_dep.index or read_dep.size != write_dep.size:
                why("epilogue's read and write indices differ")
                return False

            # the epilogue depends on expressions which may not available in the user triton kernel
            # (e.g. indexing exprs used not in a load)
            node2_inner_fn_free_symbols = node2.node.data.inner_fn_free_symbols()
            for symbol in node2_inner_fn_free_symbols:
                usages = node2.node.data.collect_inner_fn_symbol_usage(symbol)
                if any(usage != "load" for usage in usages):
                    return False

            # should be true now because we checked `can_fuse_epilogue`
            if len(node1.node.mutable_args) != 1:
                raise AssertionError(
                    f"expected one mutable arg, got {len(node1.node.mutable_args)}"
                )

            # Compare layouts, modulo dtype. We allow casting during codegen.
            layout1 = node1.node.mutable_args[0].layout
            layout2 = node2.node.layout
            if not (isinstance(layout1, ir.Layout) and isinstance(layout2, ir.Layout)):
                raise AssertionError("expected layout1 and layout2 to be ir.Layout")
            if (
                layout1.size != layout2.size
                or layout1.stride != layout2.stride
                or layout1.device != layout2.device
            ):
                why("node1 and node2 uses different buf layouts")
                return False

            def _is_other_node_that_references_mutation_buffer(
                other_node: BaseSchedulerNode,
            ):
                # [FUNCTION DOC] HELPER: True if other_node is neither node1 nor node2 and uses the buffer mutated by the user Triton kernel.
                return (
                    (other_node is not node1)
                    and (other_node is not node2)
                    and written_buffer_name in other_node.used_buffer_names()
                )

            if any(
                _is_other_node_that_references_mutation_buffer(node)
                for node in self.nodes
            ):
                return False

        if (
            isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node2.is_template()
        ):
            why("node2 is extern or nop")
            return False

        if node2.get_operation_names() & node1.ancestors:
            why("node1 must go before node2")
            return False

        if node2.is_template():
            if not _is_prologue_fusion_enabled(node2):
                why("prologue fusion turned off")
                return False

            if node1.is_reduction() or node1.is_template():
                why("prologue fusion only supported for pointwise nodes")
                return False

            template = node2.get_template_node_or_throw()
            allowed_prologue_inps = template.get_allowed_prologue_inps()
            if not allowed_prologue_inps:
                why("template has no allowed prologue inputs")
                return False

            unsupported_prologue_args = (
                OrderedSet(inp.get_name() for inp in template.inputs)  # type: ignore[union-attr]
                - allowed_prologue_inps
            )

            if node1.get_buffer_names() & unsupported_prologue_args:
                why("prologue fusion not implemented for kernel for these inputs")
                return False

            if node1.has_aliasing_or_mutation() or (
                template.has_aliasing_or_mutation_for_prologue_fusion(node2)
            ):
                why("template prologue can only fuse functional pointwise nodes")
                return False

            prologue_nodes = node1.get_nodes()
            for node in prologue_nodes[:-1]:
                node_outs = node.get_outputs()
                for out in node_outs:
                    if not all(user.node in prologue_nodes for user in out.users):
                        why("template prologue can only fuse nodes with a single use")
                        return False

            template_snodes = (
                [node2]
                if not isinstance(node2, FusedSchedulerNode)
                else [n for n in node2.snodes if n.is_template()]
            )
            if len(template_snodes) != 1:
                raise AssertionError(
                    f"expected one template snode, got {len(template_snodes)}"
                )
            template_snode = template_snodes[0]

            if not (
                len(prologue_nodes[-1].outputs) == 1
                and len(prologue_nodes[-1].outputs[0].users) == 1
                and prologue_nodes[-1].outputs[0].users[0].node is template_snode
            ):
                why(
                    "template prologue can only fuse nodes with a single use into template"
                )
                return False

            if not self.check_prologue_fusion_heuristics_fusable(node1, node2, why):
                return False

        if node1.is_template():
            atomic_add_mutation_epilogue = _can_fuse_atomic_add_template_epilogue(
                node1, node2
            )
            backend = self.get_backend(node1.get_device())
            if (
                (node2.has_aliasing_or_mutation() and not atomic_add_mutation_epilogue)
                or (
                    node2.is_reduction()
                    and not backend.can_fuse_reduction_epilogue(node1, node2)
                )
                or not _is_epilogue_fusion_enabled(node1)
            ):
                why("template epilogue not satisfied")
                return False
            template_buf = node1.get_template_node()
            if template_buf is None:
                raise AssertionError("expected template_buf to be set")
            if template_buf.is_multi_outputs_template() and not isinstance(
                node2.node, ir.ComputedBuffer
            ):
                why("multi-output template epilogue requires ComputedBuffer")
                return False

        if (node1.get_buffer_names() & V.graph.no_fuse_buffer_names) or (
            node2.get_buffer_names() & V.graph.no_fuse_buffer_names
        ):
            why("fusion for buffer explicit disabled")
            return False
        device = node1.get_device()
        device2 = node2.get_device()
        if device != device2:
            why("device mismatch (%s vs %s)", device, device2)
            return False
        del device2

        if index_equivalent_dep_names is None:
            index_equivalent_dep_names = self._nested_index_equivalent_dep_names(
                node1, node2
            )
        shared_data_score = self._score_fusion_memory_for_can_fuse(
            node1,
            node2,
            allow_mix_order_reduction=allow_mix_order_reduction,
            index_equivalent_dep_names=index_equivalent_dep_names,
        )

        if config.expand_dimension_for_pointwise_nodes and (
            expand_analysis := self.get_expand_dim_for_pointwise_nodes(node1, node2)
        ):
            (expand_dim, smaller_node, expand_size) = expand_analysis
            smaller_node.expand_dimension_for_pointwise_node(expand_dim, expand_size)
            shared_data_score = self._score_fusion_memory_for_can_fuse(
                node1,
                node2,
                index_equivalent_dep_names=index_equivalent_dep_names,
            )

        if (
            can_reorder
            and shared_data_score < config.score_fusion_memory_threshold
            and (
                config.loop_ordering_after_fusion or config.loop_reindexing_after_fusion
            )
        ):
            new_shared_data_score = self.shared_data_after_reordering_loop(node1, node2)
            if new_shared_data_score >= 0:
                shared_data_score = new_shared_data_score

        if (
            config.loop_index_inversion_in_fusion
            and shared_data_score < config.score_fusion_memory_threshold
        ):
            new_shared_data_score = self.shared_data_after_inverting_indexing(
                node1, node2
            )
            if new_shared_data_score >= 0:
                shared_data_score = new_shared_data_score

        if loop_ordering_log.isEnabledFor(logging.DEBUG):
            loop_ordering_log.debug(
                "%s and %s has %s shared data",
                node1.get_name(),
                node2.get_name(),
                shared_data_score,
            )

        if not V.choices.can_fuse(self, node1, node2, shared_data_score):
            return False

        if node1.get_operation_names() & node2.ancestors:
            # node2 depends on node1 outputs
            if (
                self.can_fuse_vertical(
                    node1,
                    node2,
                    index_equivalent_dep_names=index_equivalent_dep_names,
                )
                and V.choices.can_fuse_vertical(self, node1, node2, shared_data_score)
                and self.get_backend(device).can_fuse_vertical(node1, node2)
            ):
                return True

            # Vertical fusion failed — the iteration domains may not
            # match (e.g. pointwise reads buf[x//32] while reduction
            # writes buf[x]).  Try reindexing the pointwise to the
            # reduction's domain and retry.
            if (
                config.loop_reindexing_after_fusion
                and self._try_reindex_pointwise_for_reduction(node1, node2)
            ):
                return (
                    self.can_fuse_vertical(
                        node1,
                        node2,
                        index_equivalent_dep_names=index_equivalent_dep_names,
                    )
                    and V.choices.can_fuse_vertical(
                        self, node1, node2, shared_data_score
                    )
                    and self.get_backend(device).can_fuse_vertical(node1, node2)
                )

            return False
        else:  # nodes don't depend on each other, but may have common reads
            return V.choices.can_fuse_horizontal(
                self, node1, node2, shared_data_score
            ) and self.get_backend(device).can_fuse_horizontal(node1, node2)

    def can_fuse_vertical(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        *,
        index_equivalent_dep_names: OrderedSet[str] | None = None,
    ) -> bool:
        # [FUNCTION DOC] Checks that consumer node2 can be fused into producer node1: every unmet dependency of node2 on node1 must match a write of node1 (by exact, normalized or optionally index equivalent access) or a fusable
        # weak dependency, and no other dependency may be produced by nodes lying between node1 and node2 in the graph. Returns bool, recording reasons via WhyNoFuse.
        """
        Check if it is legal to fuse a consumer (node2) into a producer (node1).

        We can fuse them if all the reads of node2 either match
        corresponding writes in node1, or are written by nodes that can
        be scheduled before the fusion of node1 and node2.

        ``index_equivalent_dep_names`` relaxes write/read matching only for
        named producer outputs; the remaining intermediate-dependency checks
        still run normally.
        """
        node1_buf_names = node1.get_buffer_names()
        why = WhyNoFuse(node1, node2)
        remaining_deps_by_name: dict[str, list[Dep]] = defaultdict(list)

        for dep in node2.unmet_dependencies:
            name = self.mutation_renames.get(dep.name, dep.name)
            if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
                continue
            remaining_deps_by_name[name].append(dep)

        for cd in node1.read_writes.writes:
            if not isinstance(cd, MemoryDep) and not isinstance(cd, StarDep):
                continue
            write_name = self.mutation_renames.get(cd.name, cd.name)
            remaining = remaining_deps_by_name.get(write_name)
            if remaining:
                for rd in remaining:
                    if isinstance(cd, MemoryDep) and self.fusable_read_and_write(
                        rd.rename(self.mutation_renames),
                        cd,
                        allow_index_equivalence=(
                            index_equivalent_dep_names is not None
                            and write_name in index_equivalent_dep_names
                        ),
                    ):
                        remaining.remove(rd)  # noqa: B909
                    elif isinstance(cd, StarDep) and (
                        self.fusable_stardep_write_and_read_on_empty_tensor(
                            rd, cd, node1.node
                        )
                    ):
                        remaining.remove(rd)  # noqa: B909

        remaining_deps = OrderedSet(
            dep.name
            for dep in itertools.chain.from_iterable(remaining_deps_by_name.values())
        )

        if remaining_deps & node1_buf_names:
            # MemoryDeps didn't match and read different locations of the same buffer.
            # Examples here include:
            #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
            #   - MemoryDep("foo", x) != StarDep("foo")
            why("memory deps did not match")
            return False

        node1_op_names = node1.get_operation_names()
        for name in remaining_deps:
            op_name = self.name_to_buf[name].defining_op_name()
            if node1_op_names & self.name_to_fused_node[op_name].ancestors:
                why("intermediate nodes between node1 & node2")
                return False

        return True

    def fusable_weak_dep(
        self, weak_dep: WeakDep, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] True if a weak (write-after-read ordering) dependency of node2 on node1 can be dropped by fusing: node1 must own the buffer, node2 must mutate it in place with a single memory write at the same index
        # as node1's (at most one concurrent) read, injective, without temporaries, and not on halide backends.
        if weak_dep.name not in node1.get_buffer_names():
            return False

        # A weak dep can be fused if and only if the fused operation acts inplace
        # on the buffer being mutated. i.e. the same index is being read then mutated
        mutating_writes = [
            write
            for write in node2.read_writes.writes
            if write.name == weak_dep.mutating_buf
        ]
        if len(mutating_writes) != 1:
            return False
        write = mutating_writes[0]
        if isinstance(write, StarDep):
            return False
        if not isinstance(write, MemoryDep):
            raise AssertionError("expected write to be a MemoryDep")

        if free_symbol_is_type(write.index, SymT.TMP):
            return False

        # Non-injective scatter: range vars absent from write index mean
        # multiple iterations hit the same location. Can't fuse the reader
        # in or it will see partially-written state between iterations.
        if not OrderedSet(write.var_names) <= write.index.free_symbols:
            return False

        real_name = self.mutation_real_name[weak_dep.mutating_buf]
        relevant_reading_nodes = [node1]
        if isinstance(node1, ForeachKernelSchedulerNode):
            relevant_reading_nodes = node1.snodes
        num_concurrent_reads = 0
        for reading_node in relevant_reading_nodes:
            relevant_reads = [
                read
                for read in reading_node.read_writes.reads
                if read.name == real_name
            ]
            if not relevant_reads:
                continue
            device = node2.get_device()
            if device is not None and (
                (device.type == "cpu" and config.cpu_backend == "halide")
                or (device.type == "cuda" and config.cuda_backend == "halide")
            ):
                # Halide autoschedules may overcompute output tiles via
                # TailStrategy::ShiftInwards.  That is only semantics-preserving
                # if the output is not also an input read by the fused producer.
                return False
            num_concurrent_reads += 1
            if not all(
                isinstance(read, MemoryDep)
                and not free_symbol_is_type(read.index, SymT.TMP)
                and read.index == write.index
                and read.size == write.size
                for read in relevant_reads
            ):
                return False
        return num_concurrent_reads <= 1

    @staticmethod
    def _same_index_with_prefix_size(read: MemoryDep, write: MemoryDep) -> bool:
        # [FUNCTION DOC] Static helper: True if the read and write have the same index expression and the read sizes start with the write sizes.
        return (
            read.index == write.index
            and len(read.size) >= len(write.size)
            and read.size[: len(write.size)] == write.size
        )

    # StarDep doesn't match MemoryDep, and indirect indexing is not fusible.
    def fusable_read_and_write(
        self, read: Dep, write: MemoryDep, *, allow_index_equivalence: bool = False
    ) -> bool:
        # [FUNCTION DOC] Returns whether a producer write satisfies a consumer read. Memory deps match by exact index and size prefix (also after loop normalization when loop ordering after fusion is on); writes with
        # synchronizing modes never match; temporaries never match; with allow_index_equivalence relaxed consumer-side equivalences (broadcast, reshape) are accepted. A StarDep read matches a write with the same buffer name and
        # non-None mode.
        """Return whether a producer write can satisfy a consumer read.

        The default path accepts exact matches, plus the existing
        loop-ordering-normalized exact match when that config is enabled.
        ``allow_index_equivalence`` only runs after those checks fail. It keeps
        the producer write dense and injective, then accepts conservative
        consumer-side equivalent reads such as broadcasts or normalized
        loop-order changes.
        """
        if isinstance(read, MemoryDep):
            if (
                read.name != write.name
                or free_symbol_is_type(read.index, SymT.TMP)
                or free_symbol_is_type(write.index, SymT.TMP)
            ):
                return False

            original_read = read
            original_write = write

            # Operations like index_add_, scatter_add_, etc. require global
            # synchronization - all threads must complete writes before any reads.
            # These cannot be safely fused into the same kernel. Atomic modes
            # and TMA stores require synchronization barriers.
            if self.mode_requires_synchronization(original_write.mode):
                return False

            # Preserve the normal exact-dependency path before any optional
            # normalization or relaxed equivalence checks.
            if self._same_index_with_prefix_size(original_read, original_write):
                return True

            if config.loop_ordering_after_fusion and read.num_vars != write.num_vars:
                # Need merge loops if we do loop ordering after fusion since
                # we have not merged the loops yet when creating the scheduler
                # nodes.
                read = read.normalize()
                write = write.normalize()

            # Re-check after optional loop normalization. The first exact
            # match preserves original deps, including gapped layouts; this
            # path only covers deps that become exact after loop vars merge.
            if self._same_index_with_prefix_size(read, write):
                return True

            if not allow_index_equivalence:
                return False

            return self._fusable_read_after_index_equivalence(
                original_read, original_write
            )
        elif isinstance(read, StarDep):
            if (
                read.mode == write.mode
                and write.mode is not None
                and read.name == write.name
            ):
                return True
        return False

    def _fusable_read_after_index_equivalence(
        self, read: MemoryDep, write: MemoryDep
    ) -> bool:
        # [FUNCTION DOC] HELPER: Relaxed matching for a read against a producer write: requires the write to be injective and dense (contiguous after normalization), then accepts reads equal after normalization or after a
        # conservative broadcast match.
        # Relaxed matching is only for consumer-side reshapes/broadcasts.
        # If a write var is absent from the write index, the producer itself
        # broadcasts multiple loop iterations to the same address.
        if not OrderedSet(write.var_names) <= write.index.free_symbols:
            return False
        # Once read/write indices differ, require the producer to write a
        # dense logical region. Otherwise gaps or aliases in the producer
        # could be hidden by a consumer-side broadcast.
        if not (
            write.normalize().is_contiguous()
            or write.normalize_with_stride_order().is_contiguous()
        ):
            return False

        return self.deps_match_normalized(
            read, write
        ) or self._fusable_read_after_broadcast(read, write)

    @staticmethod
    def _fusable_read_after_broadcast(read: MemoryDep, write: MemoryDep) -> bool:
        # [FUNCTION DOC] HELPER: Static. Matches reads that broadcast a dense producer write: (1) after dropping read loop variables that do not influence the address the normalized read equals the normalized write, or (2) a read
        # axis that is a multiple of the write axis splits into (write axis, tail) and the tail disappears from the simplified index. Returns False otherwise.
        """Match conservative broadcasted read forms.

        This handles two nested-reduction dependency shapes:

        - Pure broadcast dims absent from the read index:

              read:  d1, {d0: 1024, d1: 16}
              write: d0, {d0: 16}

        - Same-rank expanded dims used through a quotient:

              read:  32*d0 + FloorDiv(d1, 128), {d0: 128, d1: 4096}
              write: 32*d0 + d1,                {d0: 128, d1: 32}

        Producer-side broadcast and non-dense writes are rejected before this
        helper, so these cases only relax consumer-side broadcasts.
        """
        # Strategy 1: remove read loop vars that do not affect the address,
        # then compare the normalized access against the producer write.
        read_vars = tuple(
            var for var in read.var_names if var in read.index.free_symbols
        )
        if len(read_vars) != read.num_vars:
            read_ranges = {var: read.ranges[var] for var in read_vars}
            read = MemoryDep(
                read.name,
                read.index,
                tuple(read_ranges),
                tuple(read_ranges.values()),
                read.mode,
            )
            if read.normalize() == write.normalize():
                return True

        if read.num_vars != write.num_vars:
            return False

        # Strategy 2: split a larger read axis into (write axis, tail), then
        # simplify with the tail range. This only matches if the tail
        # disappears from the final index.
        sizevars = V.graph.sizevars
        write_vars = tuple(
            sympy.Symbol(f"_fusable_broadcast_{i}", integer=True, nonnegative=True)
            for i in range(read.num_vars)
        )
        replacements: dict[sympy.Expr, sympy.Expr] = {}
        tail_ranges: dict[sympy.Symbol, sympy.Expr] = {}
        for read_var, read_size, write_var, write_size in zip(
            read.var_names, read.size, write_vars, write.size
        ):
            if sizevars.statically_known_equals(read_size, write_size):
                replacements[read_var] = write_var
                continue

            if not sizevars.statically_known_equals(
                sympy.Mod(read_size, write_size), 0
            ):
                return False
            factor = sizevars.simplify(FloorDiv(read_size, write_size))
            if not sizevars.statically_known_gt(factor, 1):
                return False

            tail_var = sympy.Symbol(
                f"_fusable_broadcast_tail_{len(tail_ranges)}",
                integer=True,
                nonnegative=True,
            )
            replacements[read_var] = write_var * factor + tail_var
            tail_ranges[tail_var] = factor

        if not tail_ranges:
            return False

        read_index = sizevars.simplify_with_ranges(
            sympy_subs(read.index, replacements),
            {**dict(zip(write_vars, write.size)), **tail_ranges},
        )
        write_index = sympy_subs(write.index, dict(zip(write.var_names, write_vars)))
        return sizevars.statically_known_equals(read_index, write_index)

    # on tensors that are "empty" (i.e. with undefined values),
    # we relax the conditions for fusion and additionally allow matching a writing StarDep with any read dep.
    # This makes use of the fact that `f(UB) = UB`.
    def fusable_stardep_write_and_read_on_empty_tensor(
        self, read: Dep, write: StarDep, writing_node: ir.Operation | None
    ) -> bool:
        # [FUNCTION DOC] For writes to the uninitialized output tensor of an epilogue-fusable user Triton kernel, allows any read of the same (mutation-renamed) buffer to match the write StarDep; False otherwise.
        if not isinstance(writing_node, ir.UserDefinedTritonKernel):
            return False
        if not writing_node.can_fuse_epilogue():
            return False
        read_name = self.mutation_renames.get(read.name, read.name)
        write_name = self.mutation_renames.get(write.name, write.name)
        if isinstance(write, StarDep) and read_name == write_name:
            return True
        return False

    @staticmethod
    def deps_match_normalized(dep1: Dep, dep2: Dep) -> bool:
        # [FUNCTION DOC] Static. True if both deps are memory deps with the same access pattern after normalization (exact equality, same variable count with stride order normalization, else loop merging normalization).
        """Check if two deps refer to the same access pattern after normalization.

        Handles the case where FusedSchedulerNodes have more loop vars
        than a single SchedulerNode (e.g., 3 vars vs 2) by falling back
        to normalize() which merges loops before comparing.
        """
        if not isinstance(dep1, MemoryDep) or not isinstance(dep2, MemoryDep):
            return False
        if dep1 == dep2:
            return True
        if dep1.num_vars == dep2.num_vars:
            return (
                dep1.normalize_with_stride_order() == dep2.normalize_with_stride_order()
            )
        return dep1.normalize() == dep2.normalize()

    def dep_size_hint(self, dep: Dep, count_bytes: bool = True) -> int:
        # [FUNCTION DOC] Returns the estimated size (bytes if count_bytes, else elements) of the accesses of dep, from V.graph.
        return V.graph.get_dep_size_hint(dep, count_bytes)

    def _score_fusion_memory_by_fusable_read_write(
        self,
        producer: BaseSchedulerNode,
        consumer: BaseSchedulerNode,
        count_bytes: bool = True,
        index_equivalent_dep_names: OrderedSet[str] | None = None,
    ) -> int:
        # [FUNCTION DOC] HELPER: Score for vertical producer to consumer pairs that exact dependency matching missed but fusable_read_and_write accepts (normalized or index equivalent): sums, per matched consumer read, the larger size
        # of the read and write. Returns 0 if the consumer does not depend on the producer.
        """Score vertical producer-output deps missed by exact dep scoring.

        Exact scoring is set-intersection based, but vertical legality can also
        accept normalized equivalent read/write deps. Give those pairs a memory
        score so heuristics do not discard them before legality runs.
        """
        if not (producer.get_operation_names() & consumer.ancestors):
            return 0

        reads = list(consumer.read_writes.reads)
        matched_reads: OrderedSet[int] = OrderedSet()
        score = 0
        for write in producer.read_writes.writes:
            if not isinstance(write, MemoryDep):
                continue
            write_name = self.mutation_renames.get(write.name, write.name)
            for i, read in enumerate(reads):
                if i in matched_reads:
                    continue
                if self.fusable_read_and_write(
                    read.rename(self.mutation_renames),
                    write,
                    allow_index_equivalence=(
                        index_equivalent_dep_names is not None
                        and write_name in index_equivalent_dep_names
                    ),
                ):
                    matched_reads.add(i)
                    score += max(
                        self.dep_size_hint(read, count_bytes),
                        self.dep_size_hint(write, count_bytes),
                    )
                    break
        return score

    def _score_fusion_memory_for_can_fuse(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        allow_mix_order_reduction: bool = True,
        index_equivalent_dep_names: OrderedSet[str] | None = None,
    ) -> int:
        # [FUNCTION DOC] HELPER: Fusion memory score used by can_fuse: score_fusion_memory (must be an int, else AssertionError); when zero and index equivalent names are given, falls back to the fusable read/write score.
        score = self.score_fusion_memory(
            node1,
            node2,
            allow_mix_order_reduction=allow_mix_order_reduction,
        )
        if not isinstance(score, int):
            raise AssertionError(f"expected score to be int, got {type(score)}")
        if score == 0 and index_equivalent_dep_names:
            score = self._score_fusion_memory_by_fusable_read_write(
                node1,
                node2,
                index_equivalent_dep_names=index_equivalent_dep_names,
            )
        return score

    @overload
    def score_fusion_memory(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        count_bytes: bool = ...,
        return_is_mix_order_reduction: Literal[False] = ...,
        allow_mix_order_reduction: bool = ...,
    ) -> int: ...

    @overload
    def score_fusion_memory(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        count_bytes: bool = ...,
        return_is_mix_order_reduction: Literal[True] = ...,
        allow_mix_order_reduction: bool = ...,
    ) -> tuple[int, int, bool]: ...

    def score_fusion_memory(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        count_bytes: bool = True,
        return_is_mix_order_reduction: bool = False,
        allow_mix_order_reduction: bool = True,
    ) -> int | tuple[int, int, bool]:
        # [FUNCTION DOC] First term of the fusion score: estimated saved memory traffic (bytes or elements per count_bytes). Mix order reduction pairs score by the numel of node1. For user Triton kernels with epilogue fusion or
        # templates it sums, over name-matched deps of the two nodes, the larger dep size. Otherwise sums the sizes of identical shared deps; if that is zero (and the pair is eligible) adds a buffer overlap score for same-buffer
        # different-index reads. With return_is_mix_order_reduction returns the tuple (score, overlap score, is mix order flag), else a single int (sum). Mix order detection can be disabled by allow_mix_order_reduction.
        """
        The first term in our fusion score that estimates number of saved
        memory operations.

        This function scores fusion candidates based on shared memory access patterns.
        Higher scores indicate better fusion candidates.

        Scoring strategy:
        1. If nodes share exact memory deps (same buffer + same indexing), return
           the sum of shared dep sizes (original behavior).
        2. If no dependency score remains, check for same-buffer reads with
           different indexing (e.g., split operations reading different slices).
           - Give bonus if nodes read from exactly the same set of buffers
           - Score based on overlap ratio: common_buffer_size / total_read_size
           - High overlap (>50%) suggests good cache locality benefit from fusion
        """

        def _construct_return_value(
            score, buffer_overlap_score, is_mix_order_reduction
        ):
            # [FUNCTION DOC] HELPER: Builds the return value: the (score, overlap score, flag) tuple when requested, else the sum of score and overlap score.
            if return_is_mix_order_reduction:
                return (score, buffer_overlap_score, is_mix_order_reduction)
            return score + buffer_overlap_score

        if allow_mix_order_reduction and MixOrderReduction.can_fuse(node1, node2):
            # The fusion score for mix order reduction only count
            # numel so far. It's actually fine. This makes other fusions
            # sharing the same amount of numels go first; but make
            # fusions only share weight/bias go later.
            score = MixOrderReduction.get_fusion_score(node1, node2)
            return _construct_return_value(score, 0, True)

        # For UserDefinedTritonKernel, the write deps are StarDep that won't
        # match the epilogue's MemoryDep via set intersection.  For templates,
        # view/reshape between the template output and epilogue can produce
        # different index expressions that don't match via set intersection.
        # Fall back to name-based matching so that the fusion score reflects
        # the actual shared buffers.
        if (
            (
                isinstance(node1.node, ir.UserDefinedTritonKernel)
                and node1.node.can_fuse_epilogue()
            )
            or node1.is_template()
            or node2.is_template()
        ):
            node1_deps = node1.read_writes.reads | node1.read_writes.writes
            node2_deps = node2.read_writes.reads | node2.read_writes.writes

            def _match(dep1: Dep, dep2: Dep):
                # [FUNCTION DOC] HELPER: True if the two deps are equal, or both are star/memory deps on the same buffer name.
                if dep1 == dep2:
                    return True
                if isinstance(dep1, (StarDep, MemoryDep)) and isinstance(
                    dep2, (StarDep, MemoryDep)
                ):
                    return dep1.name == dep2.name
                return False

            score = 0
            for node1_dep in node1_deps:
                for node2_dep in node2_deps:
                    if _match(node1_dep, node2_dep):
                        score += max(
                            self.dep_size_hint(node1_dep), self.dep_size_hint(node2_dep)
                        )

            return _construct_return_value(score, 0, False)

        node1_deps = node1.read_writes.reads | node1.read_writes.writes
        node2_deps = node2.read_writes.reads | node2.read_writes.writes
        if len(node1_deps) > len(node2_deps):
            node1_deps, node2_deps = node2_deps, node1_deps

        common_memory_deps = [dep for dep in node1_deps if dep in node2_deps]
        score = sum(self.dep_size_hint(dep, count_bytes) for dep in common_memory_deps)
        if score:
            return _construct_return_value(score, 0, False)

        # This is a horizontal/common-read locality heuristic, not a
        # producer-consumer dependency match.
        buffer_overlap_score = 0
        if score == 0 and self._can_use_buffer_overlap_scoring(node1, node2):
            buffer_overlap_score = self._score_fusion_memory_by_buffer_overlap(
                node1, node2
            )

        return _construct_return_value(score, buffer_overlap_score, False)

    def _can_use_buffer_overlap_scoring(
        self,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> bool:
        # [FUNCTION DOC] HELPER: Whether buffer overlap scoring applies: not for reductions or templates, and, under max-autotune, not when both nodes are prologue-fusion candidates for the same template (horizontal fusion
        # would block absorbing them into the template) or when a node reads the output of an epilogue-fusable template.
        """
        Check if buffer overlap scoring should be used for this node pair.

        Buffer overlap scoring handles split/cat patterns where nodes read from
        the same buffer at different indices. We skip it when:
        - Either node is a reduction (different memory access patterns)
        - Either node is a template
        - Both nodes are prologue/epilogue candidates for the same template,
          because horizontal fusion would prevent them from being absorbed
          into the template kernel. For example, in:
            q = a[:64, :]; k = a[64:, :]
            return mm(q + 2, k - 2)
          "q + 2" and "k - 2" both read from `a` and would get a high overlap
          score, but fusing them horizontally prevents prologue fusion into mm
          (resulting in 2 kernels instead of 1).

        We allow buffer overlap scoring when:
        - The node outputs are not actually in the template's allowed_prologue_inps,
          meaning they can't be prologue-fused anyway, so horizontal fusion doesn't
          prevent any optimization opportunity.
        """
        if node1.is_reduction() or node2.is_reduction():
            return False
        if node1.is_template() or node2.is_template():
            return False

        if config.max_autotune or config.max_autotune_gemm:
            node1_outputs = node1.get_outputs()
            node2_outputs = node2.get_outputs()

            # Early return if either node has no outputs
            if not node1_outputs or not node2_outputs:
                return True

            node1_output_names = OrderedSet(buf.get_name() for buf in node1_outputs)
            node2_output_names = OrderedSet(buf.get_name() for buf in node2_outputs)

            # Find templates that consume node1's outputs via buffer users
            # and check if the outputs are actually prologue-fusable
            node1_prologue_eligible_template_users: OrderedSet[BaseSchedulerNode] = (
                OrderedSet()
            )
            for buf in node1_outputs:
                for user in buf.users:
                    if (
                        isinstance(user.node, BaseSchedulerNode)
                        and user.node.is_template()
                        and _is_prologue_fusion_enabled(user.node)
                    ):
                        # Check if this output is actually in the template's
                        # allowed_prologue_inps. If not, fusing horizontally
                        # won't prevent any prologue fusion opportunity.
                        template_node = user.node.get_template_node()
                        if template_node is not None and isinstance(
                            template_node, ir.TritonTemplateBuffer
                        ):
                            allowed_inps = template_node.get_allowed_prologue_inps()
                            if node1_output_names & allowed_inps:
                                node1_prologue_eligible_template_users.add(user.node)
                        else:
                            # Conservative: assume it could be a prologue candidate
                            node1_prologue_eligible_template_users.add(user.node)

            # Check if any of node1's prologue-eligible template users also consume
            # node2's outputs in a prologue-eligible way
            if node1_prologue_eligible_template_users:
                for buf in node2_outputs:
                    for user in buf.users:
                        if (
                            isinstance(user.node, BaseSchedulerNode)
                            and user.node.is_template()
                            and user.node in node1_prologue_eligible_template_users
                        ):
                            # Also verify node2's output is prologue-eligible
                            template_node = user.node.get_template_node()
                            if template_node is not None and isinstance(
                                template_node, ir.TritonTemplateBuffer
                            ):
                                allowed_inps = template_node.get_allowed_prologue_inps()
                                if node2_output_names & allowed_inps:
                                    return False
                            else:
                                # Conservative: block fusion
                                return False

            for node in (node1, node2):
                for dep in node.read_writes.reads:
                    producer = self.name_to_fused_node.get(dep.name)
                    if (
                        producer is not None
                        and producer.is_template()
                        and _is_epilogue_fusion_enabled(producer)
                    ):
                        return False

        return True

    def _score_fusion_memory_by_buffer_overlap(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        # [FUNCTION DOC] HELPER: Scores horizontal fusion by overlap of read buffers (e.g. split slices of one buffer) when exact dependency matching finds nothing. Sums sizes of all read deps per node (using a fallback size of 10
        # when the size hint is zero), takes the larger common-buffer read size and returns it if its ratio to the larger total read size is at least config.min_overlap_ratio, else 0. Returns 0 with no common buffer.
        # KNOWN ISSUE: the docstring mentions a high bonus when both nodes read exactly the same buffers, but the code has no such bonus.
        """
        Score fusion based on buffer name overlap when exact dep matching fails.

        This handles the split/cat fusion case where nodes read from the same buffer
        but at different indices (e.g., different slices from a split operation).

        Scoring logic:
        - If nodes read from exactly the same buffers: high bonus (encourages fusion)
        - For common buffers: score based on overlap ratio
          - overlap_ratio = common_buffer_size /
            max(node1_total_reads, node2_total_reads)
          - If overlap_ratio > threshold (e.g., 0.5): give proportional score
          - If overlap_ratio < threshold: minimal/no score (not worth fusing)

        Note on dynamic shapes:
        - When deps have unbacked symbols (dynamic shapes), dep_size_hint returns 0
        - In this case, we use count * 10 as a proxy for size
        - This ensures fusion still works for models with dynamic batch sizes

        Note on multiple deps from same buffer:
        - A node may have multiple MemoryDep entries for the same buffer name
          (e.g., 4 split reads from arg0_1 at different indices)
        - We sum ALL dep sizes for each buffer, not just take max
        - This ensures overlap ratio is calculated correctly when nodes read
          multiple slices from the same underlying buffer
        """
        # Fallback size when dep_size_hint returns 0 (e.g., unbacked symbols)
        FALLBACK_DEP_SIZE = 10

        def get_dep_size(dep: Dep) -> int:
            # [FUNCTION DOC] HELPER: Returns the size hint of a dep, or the fallback size when the hint is not positive.
            size = self.dep_size_hint(dep)
            return size if size > 0 else FALLBACK_DEP_SIZE

        node1_read_names = OrderedSet(dep.name for dep in node1.read_writes.reads)
        node2_read_names = OrderedSet(dep.name for dep in node2.read_writes.reads)

        # Early exit if no common buffer names
        common_names = node1_read_names & node2_read_names

        if not common_names:
            return 0

        # Calculate total read sizes for each node (sum of ALL deps)
        node1_total_read_size = sum(
            get_dep_size(dep) for dep in node1.read_writes.reads
        )
        node2_total_read_size = sum(
            get_dep_size(dep) for dep in node2.read_writes.reads
        )

        max_total_read_size = max(node1_total_read_size, node2_total_read_size)
        if max_total_read_size == 0:
            return 0

        # Calculate total reads from common buffers for each node
        # Sum ALL deps for each common buffer name
        # (handles multiple reads from same buffer)
        node1_common_read_size = sum(
            get_dep_size(dep)
            for dep in node1.read_writes.reads
            if dep.name in common_names
        )
        node2_common_read_size = sum(
            get_dep_size(dep)
            for dep in node2.read_writes.reads
            if dep.name in common_names
        )

        # Use max of the two as the common buffer size estimate
        # This represents how much data is being read from shared buffers
        common_read_buffer_size = max(node1_common_read_size, node2_common_read_size)

        # Calculate overlap ratio
        overlap_ratio = common_read_buffer_size / max_total_read_size
        # Scale score by overlap ratio and common buffer size
        # Higher overlap = higher score
        # Larger common buffer = higher score (more cache benefit)
        return (
            common_read_buffer_size if overlap_ratio >= config.min_overlap_ratio else 0
        )

    def get_possible_fusions_with_highest_priority(
        self, possible_fusions: list[tuple[BaseSchedulerNode, BaseSchedulerNode]]
    ) -> list[tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        # [FUNCTION DOC] Groups candidate fusion pairs by the backend's fusion pair priority (lower number is higher priority) and returns only the group with the highest priority; empty input returns unchanged. Raises AssertionError if a
        # pair's nodes are on different devices.
        # Group the possible fusions based on their priority from the backend.
        # Only return the group of possible fusions with highest priority.
        if len(possible_fusions) == 0:
            return possible_fusions
        possible_fusions_group_by_priority: dict[
            int, list[tuple[BaseSchedulerNode, BaseSchedulerNode]]
        ] = {}

        for node1, node2 in possible_fusions:
            if node1.get_device() != node2.get_device():
                raise AssertionError(
                    "expected node1 and node2 to be on the same device"
                )
            device = node1.get_device()
            fusion_pair_priority = int(
                self.get_backend(device).get_fusion_pair_priority(node1, node2)
            )
            if fusion_pair_priority not in possible_fusions_group_by_priority:
                possible_fusions_group_by_priority[fusion_pair_priority] = [
                    (node1, node2),
                ]
            else:
                possible_fusions_group_by_priority[fusion_pair_priority].append(
                    (node1, node2)
                )
        # return the possible fusions with highest priority
        possible_fusions_with_highest_priority = min(
            possible_fusions_group_by_priority.items(), key=operator.itemgetter(0)
        )[1]
        if len(possible_fusions_with_highest_priority) <= 0:
            raise AssertionError(
                "expected at least one possible fusion with highest priority"
            )
        return possible_fusions_with_highest_priority

    def score_fusion_key(
        self, nodes: tuple[BaseSchedulerNode, BaseSchedulerNode]
    ) -> Any:
        # [FUNCTION DOC] Sort key shim: delegates to the choices handler's fusion scoring for the pair.
        """
        Shim for list.sort(key=...)
        """
        return V.choices.score_fusion(self, *nodes)

    def compute_last_usage(self) -> None:
        # [FUNCTION DOC] Computes last_usage for all nodes (and nodes within fused nodes) by walking nodes in reverse and tracking buffers needed by later nodes plus graph outputs.
        """
        Populate node.last_usage recursively (also for the nodes within a FusedSchedulerNode)
        """

        future_used_buffers = OrderedSet(V.graph.get_output_names())

        for node in reversed(self.nodes):
            node.set_last_usage(future_used_buffers, self.mutation_real_name)
            future_used_buffers.update(node.last_usage)

    def free_buffers(self) -> None:
        # [FUNCTION DOC] Emits free code for buffers in buffer_names_to_free that are neither removed nor already freed: scheduler buffers that can be freed, TorchBind input objects and input storages (generator/opaque state is
        # skipped), then clears the set. Raises AssertionError if an input is not an input storage buffer. Side effect: writes wrapper code.
        """Free any buffers that are no longer needed"""
        for name in sorted(
            self.buffer_names_to_free
            - V.graph.removed_buffers
            - V.graph.wrapper_code.freed  # type: ignore[has-type]
        ):
            if name in self.name_to_buf:
                buf = self.name_to_buf[name]
                if buf.can_free():
                    V.graph.wrapper_code.codegen_free(buf.node)
            elif name in V.graph.graph_inputs:
                inp = V.graph.graph_inputs[name]
                if isinstance(inp, ir.TorchBindObject):
                    V.graph.wrapper_code.codegen_free(inp)
                elif isinstance(inp, (ir.GeneratorState, ir.OpaqueObjectState)):
                    continue
                else:
                    storage = inp.data
                    if not (
                        isinstance(storage, ir.StorageBox) and storage.is_input_buffer()
                    ):
                        raise AssertionError(
                            "expected storage to be an ir.StorageBox input buffer"
                        )
                    V.graph.wrapper_code.codegen_free(storage.data)

        self.buffer_names_to_free.clear()

    def flush(self) -> None:
        # [FUNCTION DOC] Flushes all backends and frees pending buffers.
        for backend in self.backends.values():
            backend.flush()
        self.free_buffers()

    def codegen_extern_call(
        self,
        scheduler_node: BaseSchedulerNode,
    ) -> None:
        # [FUNCTION DOC] Generates the call for an extern (or fused extern Triton) node: counts the extern call, decides in-place updates and marks outputs allocated under a temporary kernel handler, emits the call and frees
        # buffers. Raises AssertionError for other node types.
        if not isinstance(
            scheduler_node,
            (ExternKernelSchedulerNode, FusedExternTritonKernelSchedulerNode),
        ):
            raise AssertionError(
                "expected an ExternKernelSchedulerNode or FusedExternTritonKernelSchedulerNode"
            )
        # 'decide_inplace_update' stores the inplace update decisions in
        # the current kernel from where 'allocate' retrieve those decisions.
        # We have to make sure there is a non-NULL kernel handler to store
        # those inplace update decisions.
        counters["inductor"]["extern_calls"] += 1
        with V.set_kernel_handler(Kernel(increase_kernel_count=False)):
            scheduler_node.decide_inplace_update()
            scheduler_node.mark_run()
        scheduler_node.codegen(V.graph.wrapper_code)
        self.free_buffers()

    def create_backend(self, device: torch.device) -> BaseScheduling:
        # [FUNCTION DOC] Creates the scheduling backend for a device and registers device info in the graph. Raises AssertionError for GPU devices without an index, RuntimeError for unsupported device types, GPUTooOldForTriton or
        # TritonMissing when triton is unavailable.
        if is_gpu(device.type) and device.index is None:
            raise AssertionError(f"{device} should have been normalized in lowering")
        V.graph.add_device_info(device)

        device_scheduling = get_scheduling_for_device(device.type)
        if device_scheduling is None:
            raise RuntimeError(f"Unsupported device type: {device.type}")

        if not has_triton():
            if (
                device.type == "cuda"
                and (device_props := torch.cuda.get_device_properties(device)).major < 7
            ):
                raise GPUTooOldForTriton(device_props, inspect.currentframe())
            elif is_gpu(device.type) and not device.type == "mps":
                raise TritonMissing(inspect.currentframe())

        return device_scheduling(self)

    def get_backend(self, device: torch.device | None) -> BaseScheduling:
        # [FUNCTION DOC] Returns the cached scheduling backend for the device, creating it on first use. Raises AssertionError if device is None.
        if device is None:
            raise AssertionError("expected device to be set")
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    def enter_context(self, node: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] Emits a context marker in generated wrapper code for the node: finds the latest (by fx graph order) origin among its sub-nodes and enters that context. Caches fx node positions in origin_to_index.
        def get_order(n: torch.fx.Node) -> int:
            # [FUNCTION DOC] HELPER: Returns the position of the fx node in its graph, caching all positions on first use.
            if n not in self.origin_to_index:
                self.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})
            return self.origin_to_index[n]

        # Use a dict to have ordering
        origins = {
            (get_order(e), e): None
            for n in node.get_nodes()
            if n.node is not None
            for e in n.node.get_origins()
        }
        origins = list(origins.keys())
        if origins:
            _, last = max(origins, key=operator.itemgetter(0))
            V.graph.wrapper_code.enter_context(last)

    def can_buffer_be_removed_through_fusion(
        self, name: str, fused_node_names: OrderedSet[str]
    ) -> bool:
        # [FUNCTION DOC] True if the buffer is only used (weakly or by nodes within fused_node_names) by the nodes being fused and is not subject to mutation renames; False if unknown.
        try:
            users = self.name_to_buf[name].users
        except KeyError:
            return False
        return (
            all(user.is_weak or user.get_name() in fused_node_names for user in users)
            and name not in self.mutation_renames
            and name not in self.mutation_real_name
        )

    def should_partition(self, node: BaseSchedulerNode) -> str | None:
        # [FUNCTION DOC] Returns the reason the graph should be partitioned at this node, or None if the node is cudagraph-safe. Reasons include custom partition ops, all ops when cudagraphs are disabled, non-GPU ops, device copies,
        # switches, unbacked bindings, cudagraph-unsafe custom ops, cudagraph-unsafe unbacked symints, and dynamic shape ops when configured. Fused nodes check each sub-node. Raises AssertionError for a node without IR.
        """
        Return the reason why we should partition the inductor graph on this node,
        or None if the node is cudagraphable.
        """

        # Allow users to manually specify if a node should be partitioned
        # Can only do this for FallbackKernels
        ir_node = node.node
        if isinstance(ir_node, torch._inductor.ir.FallbackKernel) and (
            op := ir_node.op_overload
        ):
            op_overload_packet_name, op_overload_name = get_op_names(op)
            if (
                op_overload_packet_name in config.custom_should_partition_ops
                or op_overload_name in config.custom_should_partition_ops
            ):
                if not isinstance(op, torch._ops.OpOverload):
                    raise AssertionError("expected op to be a torch._ops.OpOverload")
                return f"custom partition op: {op_overload_name}"

        # When not using cudagraphs, keep all kernels in the `call` function
        # instead of graph partition functions, since graph partition only brings
        # benefit to cudagraph
        if (
            not torch._inductor.config.triton.cudagraphs
            and _unstable_customized_partition_wrapper.wrapper is None
        ):
            return "partition includes all ops when cudagraphs is disabled"

        if isinstance(node, FusedSchedulerNode):
            for snode in node.snodes:
                reason = self.should_partition(snode)
                if reason:
                    return reason
            return None

        if node.node is None:
            raise AssertionError("expected node.node to be set")

        if not node.is_gpu():
            return f"{node.get_device()} ops"

        if isinstance(node.node, ir.DeviceCopy):
            return "DeviceCopy ops"

        if isinstance(node.node, ir.Switch):
            return "Switch ops"

        if getattr(node.node, "unbacked_bindings", None):
            return "unbacked binding ops"

        if is_cudagraph_unsafe_op(node.node):
            return "CUDAGraph-unsafe custom ops"

        if reason := self._uses_cudagraph_unsafe_unbacked_symint(node):
            return reason

        # Partition around nodes with dynamic shapes when cudagraph_skip_dynamic_graphs is enabled
        if config.triton.cudagraph_skip_dynamic_graphs:
            if get_scheduler_node_symbol_uses(node):
                return "dynamic shape ops"

        return None

    @cache_on_self
    def _get_cudagraph_unsafe_unbacked_symints(self) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] HELPER: Cached. Collects the unbacked int/float symbols defined by fallback kernels listed in the config as cudagraph-unsafe unbacked ops.
        """
        Collect output unbacked symints from ops in config.cudagraph_unsafe_unbacked_ops.
        """
        unsafe_symints: OrderedSet[sympy.Symbol] = OrderedSet()

        if not config.cudagraph_unsafe_unbacked_ops:
            return unsafe_symints

        for node in self.nodes:
            ir_node = node.node
            if ir_node is None:
                continue

            if not isinstance(ir_node, torch._inductor.ir.FallbackKernel):
                continue

            op = ir_node.op_overload
            if op is None:
                continue

            op_overload_packet_name, op_overload_name = get_op_names(op)
            if (
                op_overload_packet_name not in config.cudagraph_unsafe_unbacked_ops
                and op_overload_name not in config.cudagraph_unsafe_unbacked_ops
            ):
                continue

            for sym in ir_node.get_unbacked_symbol_defs():
                sym = V.graph.sizevars.simplify(sym)
                if symbol_is_type(sym, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)):
                    unsafe_symints.add(sym)

        return unsafe_symints

    def _uses_cudagraph_unsafe_unbacked_symint(
        self, node: BaseSchedulerNode
    ) -> str | None:
        # [FUNCTION DOC] HELPER: Returns a reason string if the node uses a symbol defined by a cudagraph-unsafe unbacked op, else None.
        unsafe_symints = self._get_cudagraph_unsafe_unbacked_symints()
        if not unsafe_symints:
            return None

        node_symbols = get_scheduler_node_symbol_uses(node)

        for sym in node_symbols:
            simplified_sym = V.graph.sizevars.simplify(sym)
            for free_sym in simplified_sym.free_symbols:
                if free_sym in unsafe_symints:
                    return f"uses cudagraph-unsafe unbacked symint: {free_sym}"

        return None

    def get_name_to_nodes(
        self,
    ) -> dict[str, ir.IRNode | ir.TorchBindObject | sympy.Expr]:
        # [FUNCTION DOC] Returns a mapping from names to graph inputs and to the IR nodes of scheduler node outputs.
        """
        Return a mapping from name strings to the corresponding graph inputs or
        base scheduler node outputs.
        """
        name_to_node: dict[str, ir.IRNode | ir.TorchBindObject | sympy.Expr] = {}
        name_to_node.update(V.graph.graph_inputs)

        for node in self.nodes:
            for name, scheduler_buffer in node.outputs_by_name.items():
                name_to_node[name] = scheduler_buffer.node

        return name_to_node

    def compute_graph_partition_maps(
        self,
        signatures: list[GraphPartitionSignature],
    ) -> None:
        # [FUNCTION DOC] Fills V.graph.partition_maps: for each non-skipped partition signature, a GraphPartitionMap with its partition id, the graph input index for each partition input name (None if not a graph input), the graph output
        # indices for each output node and constant names (mutates V.graph).
        """
        computes a mapping from partition input/output indices to graph input/output
        indices for each partition.
        """
        name_to_graph_input_index = {
            name: idx for idx, name in enumerate(V.graph.graph_inputs)
        }
        name_to_graph_output_indices: dict[str, list[int]] = defaultdict(list)
        for idx, name in enumerate(V.graph.get_output_names()):
            name_to_graph_output_indices[name].append(idx)

        V.graph.partition_maps = []
        for partition_id, signature in enumerate(signatures):
            if signature.skip_cudagraph:
                # Note: [Graph Partition Map for CUDAGraph]
                # number of partition map should be the same as the number of generated
                # partition functions. This assumption will be used when cudagraphify
                # each partition function.
                continue

            input_mapping = []
            for name in signature.input_nodes:
                input_mapping.append(name_to_graph_input_index.get(name))

            output_mapping = []
            for node in signature.output_nodes:
                output_mapping.append(
                    name_to_graph_output_indices.get(node.get_name(), [])
                )

            V.graph.partition_maps.append(
                GraphPartitionMap(
                    partition_id,
                    input_mapping,
                    output_mapping,
                    signature.constant_names,
                )
            )

    def get_graph_partition_symbol_inputs(
        self,
        partition: PartitionType,
        input_nodes: dict[str, ir.IRNode | ir.TorchBindObject | sympy.Expr],
    ) -> OrderedSet[sympy.Symbol]:
        # [FUNCTION DOC] Returns the sorted symbols needed in scope for a partition: free symbols used by its nodes and by the layouts of its input nodes, restricted to size/float/unbacked symbols and reduced to the free symbols of
        # their simplified forms.
        """
        Returns all symbol inputs which are required to be in scope to successfully
        perform codegen for this graph partition, including:
        - free symbols used in partition nodes
        - free symbols in partition input/node shapes, strides, and offsets. This is needed
          for recording cudagraphs for tensors with dynamic shapes.
        """

        def get_input_node_symbols(
            node: ir.IRNode | sympy.Expr | ir.TorchBindObject,
        ) -> OrderedSet[sympy.Symbol]:
            # [FUNCTION DOC] HELPER: Returns the layout symbols of an input node (none for TorchBind objects); raises NotImplementedError for other types.
            """
            Gets symbols used in input node shapes, strides, and offsets.
            """
            if isinstance(node, ir.TorchBindObject):
                # TorchBindObject does not involve dynamic shapes yet
                return OrderedSet()
            elif isinstance(node, ir.IRNode):
                return get_layout_symints(node)
            else:
                # node cannot be sympy.Expr since node comes from read_writes and
                # read_writes does not contain sympy.Expr
                raise NotImplementedError(f"Unsupported input node type: {type(node)}")

        def filter_symbols(
            symbols: OrderedSet[sympy.Symbol],
        ) -> OrderedSet[sympy.Symbol]:
            # [FUNCTION DOC] HELPER: Keeps only size, float and unbacked int/float symbols, dropping kernel-internal ones.
            """
            Filters a set of symbols that are required for codegen. Skip symbols
            that are always internal to kernels, such as SymT.TMP, SymT.INDEX,
            and SymT.R0_INDEX.
            """
            return OrderedSet(
                s
                for s in symbols
                if symbol_is_type(
                    s,
                    (
                        SymT.SIZE,
                        SymT.FLOAT,
                        SymT.UNBACKED_INT,
                        SymT.UNBACKED_FLOAT,
                    ),
                )
            )

        candidate_symbols: OrderedSet[sympy.Symbol] = OrderedSet().union(
            *(get_scheduler_node_symbol_uses(node) for node in partition)
        )
        candidate_symbols.update(
            *(get_input_node_symbols(node) for node in input_nodes.values())
        )

        candidate_symbols = filter_symbols(candidate_symbols)

        res: OrderedSet[sympy.Symbol] = OrderedSet()
        for s in candidate_symbols:
            symplified_s = V.graph.sizevars.simplify(s)
            # use free_symbols only when s is simplified to an Integer or expr
            res.update(symplified_s.free_symbols)

        return OrderedSet(sorted(res, key=operator.attrgetter("name")))

    def get_graph_partition_signature(
        self, partitions: list[PartitionType], skip_cudagraphs: list[bool]
    ) -> list[GraphPartitionSignature]:
        # [FUNCTION DOC] Builds a GraphPartitionSignature for each partition (processed in reverse order, tracking outputs still needed): the partition's input nodes, output nodes, symbol inputs, constants and whether to skip
        # cudagraphs, ignoring unallocated none-layout buffers.
        """
        Gets signature for each graph partition, including input nodes, output nodes, and
        whether deallocating an input within graph partition.
        """
        signatures = []

        unmet_output_names = OrderedSet(V.graph.get_output_names())
        name_to_node = self.get_name_to_nodes()

        def is_unallocated_buffer(buf_name: str) -> bool:
            # [FUNCTION DOC] HELPER: True if the buffer (following mutation real names) has none layout and so is not allocated.
            """
            Checks if buf_name resolves to a NoneLayout buffer (following mutation_real_name).
            Buffers with NoneLayout are not allocated so graph partition should not
            take them as inputs or outputs.
            """
            buf = self.name_to_buf.get(buf_name, None)

            if buf is None:
                return False

            if isinstance(buf.node.layout, NoneLayout):
                # If there's a mutation real name, check the underlying buffer
                # This handles both MutationOutput and other mutation ops like
                # IndexPutFallback that have NoneLayout but mutate real buffers
                if real_name := self.mutation_real_name.get(buf_name, None):
                    return is_unallocated_buffer(real_name)

                return True

            return False

        for partition, skip_cudagraph in zip(
            reversed(partitions), reversed(skip_cudagraphs)
        ):
            output_names: OrderedSet[str] = OrderedSet()

            for node in partition:
                output_names.update(node.outputs_by_name.keys())

            returned_output_names = output_names.intersection(unmet_output_names)

            # all reads/writes are partition inputs except those generated
            # within the partition and tensor constants
            read_writes = dependencies.ReadWrites.merge_list(
                [node.read_writes for node in partition]
            )

            # WeakDep is fake dependency on unused buffer. It should not appear
            # in partition_input_names for inputs that are actually read or written.
            partition_input_names = (
                _real_dep_names(read_writes.reads | read_writes.writes) - output_names
            )

            partition_input_names = OrderedSet(
                self.mutation_real_name.get(name, name)
                for name in partition_input_names
            )

            buffer_names_to_free: OrderedSet[str] = OrderedSet()
            for node in partition:
                buffer_names_to_free.update(node.last_usage)

            # buffer_names_to_free may contain buffers allocated in previous
            # graph partitions. These buffers should also be a partition
            # input.
            extra_input_names = [
                name
                for name in (buffer_names_to_free - output_names)
                if name in name_to_node
            ]
            partition_input_names.update(extra_input_names)

            input_nodes = {
                name: name_to_node[name]
                for name in partition_input_names
                if name in name_to_node
            }
            input_deallocation = {
                name: name in buffer_names_to_free
                for name in partition_input_names
                if name in name_to_node
            }

            # if an input tensor is not freed in the partition function, it should
            # also be returned as an output. This brings benefits to cudagraph
            # since the returned output tensor is a cudagraph managed tensor with
            # a static tensor address.
            extra_output_names = [
                name
                for name in partition_input_names
                if name in name_to_node and name not in buffer_names_to_free
            ]

            returned_output_names.update(extra_output_names)

            returned_output_names = OrderedSet(
                self.mutation_real_name.get(name, name)
                for name in returned_output_names
            )

            output_nodes = [
                name_to_node[name]
                for name in returned_output_names
                if not is_unallocated_buffer(name)
            ]

            constant_names = [
                name for name in partition_input_names if name in V.graph.constants
            ]

            symbol_inputs = self.get_graph_partition_symbol_inputs(
                partition, input_nodes
            )

            partition_signature = GraphPartitionSignature(
                symbol_inputs,
                input_nodes,
                output_nodes,
                input_deallocation,
                skip_cudagraph,
                constant_names,
            )

            signatures.append(partition_signature)

            unmet_output_names = partition_input_names.union(
                unmet_output_names - returned_output_names
            )

        return signatures[::-1]

    def clean_removed_buffer_from_partition_signatures(
        self,
        signature: GraphPartitionSignature,
        removed_buffers: OrderedSet[str],
    ) -> GraphPartitionSignature:
        # [FUNCTION DOC] Returns a new partition signature without the buffers listed in removed_buffers (from inputs, input deallocation flags, output nodes and constants), keeping symbol inputs and the skip flag.
        """
        Updates the partition signature by removing buffers specified in
        removed_buffers. See [Note: Removed Graph Partition Arguments]
        """
        input_nodes = {
            name: buffer
            for name, buffer in signature.input_nodes.items()
            if name not in removed_buffers
        }
        input_deallocation = {
            name: val
            for name, val in signature.input_deallocation.items()
            if name not in removed_buffers
        }
        output_nodes = [
            node
            for node in signature.output_nodes
            if node.maybe_get_name() not in removed_buffers
        ]
        constant_names = [
            name for name in signature.constant_names if name not in removed_buffers
        ]
        return GraphPartitionSignature(
            signature.symbol_inputs,
            input_nodes,
            output_nodes,
            input_deallocation,
            signature.skip_cudagraph,
            constant_names,
        )

    def reorder_for_minimizing_partition(
        self,
        nodes: list[BaseSchedulerNode],
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Reorders nodes to minimize graph partitions with a breadth first topological sort that alternates between runs of non-cudagraphable and cudagraphable nodes, each run taken in original index order
        # (priority queues by original index). Uses predecessor/successor info on the nodes. Preserves relative order within each kind. Returns the new schedule.
        # KNOWN ISSUE: the RuntimeError for an overlong loop is unreachable because the loop stops once the iteration count reaches the number of nodes and never exceeds it.
        """
        Reorder nodes to minimize the number of partitions via a bfs
        topological sort. This is the optimal reordering such that the
        number of partitions cannot be reduced further. This may be
        sub-optimal for other metrics such as peak memory. This does not
        change relative orders of two cudagraphable nodes, nor the
        relative order of two non_cudagraphable nodes.
        """
        import heapq

        node_to_indegree: dict[BaseSchedulerNode, int] = dict()
        cudagraphable_nodes: list[tuple[int, BaseSchedulerNode]] = []
        non_cudagraphable_nodes: list[tuple[int, BaseSchedulerNode]] = []
        node_to_index = {node: idx for idx, node in enumerate(nodes)}

        def insert_pending_nodes(node: BaseSchedulerNode) -> None:
            # [FUNCTION DOC] HELPER: Pushes a ready node onto the non-cudagraphable or cudagraphable queue (keyed by original index) depending on should_partition.
            node_with_index = (node_to_index[node], node)
            if self.should_partition(node):
                heapq.heappush(non_cudagraphable_nodes, node_with_index)
            else:
                heapq.heappush(cudagraphable_nodes, node_with_index)

        def update_indegree(node: BaseSchedulerNode) -> None:
            # [FUNCTION DOC] HELPER: Decrements the indegree of each successor of the node, queueing those that become ready; raises AssertionError if an indegree is not positive.
            for succ_node in node.mpi_node.succ_nodes:
                if node_to_indegree[succ_node] <= 0:
                    raise AssertionError(
                        f"expected positive indegree, got {node_to_indegree[succ_node]}"
                    )
                node_to_indegree[succ_node] -= 1
                if node_to_indegree[succ_node] == 0:
                    insert_pending_nodes(succ_node)

        for node in nodes:
            node_to_indegree[node] = len(node.mpi_node.pred_nodes)
            if node_to_indegree[node] == 0:
                insert_pending_nodes(node)

        schedule: list[BaseSchedulerNode] = []
        num_iters: int = 0
        while num_iters < len(nodes) and (
            non_cudagraphable_nodes or cudagraphable_nodes
        ):
            while non_cudagraphable_nodes:
                _, node = heapq.heappop(non_cudagraphable_nodes)
                schedule.append(node)
                update_indegree(node)

            while cudagraphable_nodes:
                _, node = heapq.heappop(cudagraphable_nodes)
                schedule.append(node)
                update_indegree(node)

            num_iters += 1

        if num_iters > len(nodes):
            raise RuntimeError(
                """
                Failed to schedule, while loop ran too long when
                reordering for minimizing the num of partitions
                """
            )

        return schedule

    def maybe_reorder_for_minimizing_partition(
        self,
        nodes: list[BaseSchedulerNode],
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Reorders nodes to minimize partitions only if the estimated peak memory of the reordered schedule stays below the original peak times the configured cudagraph partition memory budget; otherwise returns
        # the nodes unchanged.
        """
        Reorder nodes to minimize the number of partitions if this only slightly
        increase peak memory.
        """
        from .memory import estimate_peak_memory, prepare_planning_info

        graph_outputs = OrderedSet(V.graph.get_output_names())

        default_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
            nodes,
            self.name_to_buf,
            self.name_to_fused_node,
            OrderedSet(V.graph.graph_inputs.keys()),
            graph_outputs,
        )

        reordered_nodes = self.reorder_for_minimizing_partition(nodes)
        reorder_peak_memory, _ = estimate_peak_memory(
            reordered_nodes, name_to_freeable_input_buf, graph_outputs
        )

        if (
            reorder_peak_memory
            < default_peak_memory * config.triton.cudagraph_partition_memory_budget
        ):
            return reordered_nodes

        return nodes

    def reorder_for_partition_with_simple_dependency(
        self, nodes: list[BaseSchedulerNode]
    ) -> list[BaseSchedulerNode]:
        # [FUNCTION DOC] Reorders nodes that must be partitioned and have simple dependencies: moves those without unmet dependencies to the front and those used only by graph outputs to the back; others keep their order.
        """
        Reorder a node if it should be partitioned and has simple dependency:
        1. move a partitioned node to the front if it has no dependency
        2. move a partitioned node to the back if it is only used by OutputNode
        3. otherwise do not reorder
        """

        front: list[BaseSchedulerNode] = []
        middle: list[BaseSchedulerNode] = []
        back: list[BaseSchedulerNode] = []

        def only_output_user(node: BaseSchedulerNode) -> bool:
            # [FUNCTION DOC] HELPER: True if all users of the node's outputs are OutputNodes.
            for buf in node.get_outputs():
                for use in buf.users:
                    if not isinstance(use.node, OutputNode):
                        return False
            return True

        for node in nodes:
            should_partition = self.should_partition(node) is not None
            if should_partition and len(node.unmet_dependencies) == 0:
                front.append(node)
            elif should_partition and only_output_user(node):
                back.append(node)
            else:
                middle.append(node)

        return front + middle + back

    def graph_partition(
        self,
    ) -> tuple[list[PartitionType], list[GraphPartitionSignature]]:
        # [FUNCTION DOC] Splits self.nodes into consecutive partitions of cudagraph-safe or unsafe nodes (by should_partition), marks cudagraph partitions below config.triton.cudagraph_min_partition_size kernels as skipped, computes
        # signatures and partition maps (side effect on V.graph), logs, and returns (partitions, signatures).
        """
        Given a list of BaseSchedulerNodes, split into a list of
        graph partitions and compute partition input/output signatures.
        """
        partitions: list[PartitionType] = []
        skip_cudagraph = True
        cur_partition: PartitionType = []
        skip_cudagraphs = []
        for node in self.nodes:
            node_should_partition = self.should_partition(node) is not None
            if cur_partition and skip_cudagraph != node_should_partition:
                partitions.append(cur_partition)
                skip_cudagraphs.append(skip_cudagraph)
                cur_partition = []

            skip_cudagraph = node_should_partition
            cur_partition.append(node)

        if cur_partition:
            partitions.append(cur_partition)
            skip_cudagraphs.append(skip_cudagraph)

        # Apply minimum partition size threshold: if a cudagraph-eligible partition
        # has fewer kernels than the threshold, mark it as non-cudagraphable
        min_size = config.triton.cudagraph_min_partition_size
        if min_size > 0:
            for i, (partition, skip) in enumerate(zip(partitions, skip_cudagraphs)):
                if not skip:
                    kernel_count = self.count_kernel_nodes(partition)
                    if kernel_count < min_size:
                        skip_cudagraphs[i] = True
                        cudagraphs_log.debug(
                            "Partition %d has %d kernels, below minimum size %d, skipping cudagraph",
                            i,
                            kernel_count,
                            min_size,
                        )

        signatures = self.get_graph_partition_signature(
            partitions=partitions, skip_cudagraphs=skip_cudagraphs
        )
        self.compute_graph_partition_maps(signatures)

        self._log_graph_partitions(partitions, signatures)

        return partitions, signatures

    def _log_graph_partitions(
        self,
        partitions: list[PartitionType],
        signatures: list[GraphPartitionSignature],
    ) -> None:
        # [FUNCTION DOC] HELPER: Debug logs the number of cudagraphable and non-cudagraphable partitions with details and reasons, only when debug logging is enabled and a GPU device is in use.
        if not cudagraphs_log.isEnabledFor(logging.DEBUG):
            return

        # Don't log partition reasons for CPU-only graphs since cudagraph
        # partitioning is not relevant when there are no GPU devices
        has_gpu_device = any(is_gpu(device) for device in V.graph.device_types)
        if not has_gpu_device:
            return

        cudagraphable_count = sum(1 for s in signatures if not s.skip_cudagraph)
        non_cudagraphable_count = len(signatures) - cudagraphable_count
        cudagraphs_log.debug(
            "Created %d graph partitions: %d cudagraphable, %d non-cudagraphable",
            len(partitions),
            cudagraphable_count,
            non_cudagraphable_count,
        )
        for i, (partition, signature) in enumerate(zip(partitions, signatures)):
            cudagraphs_log.debug(
                "  Partition %d: %d nodes, %s, inputs=%d, outputs=%d",
                i,
                len(partition),
                "non-cudagraphable" if signature.skip_cudagraph else "cudagraphable",
                len(signature.input_nodes),
                len(signature.output_nodes),
            )
            if signature.skip_cudagraph:
                # Log details for each non-cudagraphable node
                for node in partition:
                    self._log_non_cudagraphable_node(node)

    def _log_non_cudagraphable_node(self, node: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] HELPER: Debug logs the reason, IR type, fx node and stack trace of a non-cudagraphable node, if it has a partition reason.
        """Log details for a non-cudagraphable node."""
        reason = self.should_partition(node)
        if not reason:
            return

        node_name = node.get_name()
        fx_node = node.node.get_origin_node() if node.node is not None else None
        parts = [f"reason={reason}"]
        ir_type = type(node.node).__name__
        parts.append(f"ir={ir_type}")
        if fx_node is not None:
            fx_str = f"{fx_node.target}({', '.join(str(a) for a in fx_node.args)})"
            parts.append(f"fx={fx_str}")

        cudagraphs_log.debug("    %s: %s", node_name, ", ".join(parts))

        # Log full stack trace if available
        if fx_node is not None:
            stack_trace = fx_node.meta.get("stack_trace", None)
            if stack_trace:
                for line in stack_trace.strip().split("\n"):
                    cudagraphs_log.debug("         %s", line)

    def codegen(self) -> None:
        # [FUNCTION DOC] Generates the code for the whole schedule under a timing context: per partition if graph partition is enabled, else all nodes at once.
        with dynamo_timed("Scheduler.codegen"):
            return (
                self._codegen_partitions()
                if torch._inductor.config.graph_partition
                else self._codegen(self.nodes)
            )

    def _codegen_partition_wrapper(
        self,
        partition: PartitionType,
        signature: GraphPartitionSignature,
    ) -> None:
        # [FUNCTION DOC] HELPER: Generates a cudagraph partition as a separate subgraph wrapper: creates a subgraph wrapper, codegens the partition nodes, removes buffers eliminated during codegen from the signature, writes the
        # prefix, defines the launcher function in the parent wrapper and emits the partition call, recording the outputs as allocated. Mutates V.graph wrapper code state.
        """Codegen a partition given its inputs/outputs"""
        from .codegen.wrapper import SubgraphPythonWrapperCodegen

        parent_wrapper_code = V.graph.wrapper_code
        graph_partition_id = next(self._graph_partition_counter)
        graph_name = parent_wrapper_code.get_partition_name(graph_partition_id)

        with V.graph.set_current_wrapper_code():
            V.graph.init_wrapper_code(
                is_subgraph=True,
                subgraph_name=graph_name,
                parent_wrapper_code=parent_wrapper_code,
                partition_signatures=signature,
            )
            # V.graph.removed_buffers is global to the wrapper. Only buffers added
            # while codegening this partition should be removed from this signature.
            removed_buffers_before_codegen = V.graph.removed_buffers.copy()
            self._codegen(partition)

            # Note: [Removed Graph Partition Arguments]
            # Graph partition relies on node.read_writes to analyze the partition
            # inputs and outputs. However, during codegen, we may decide some buffers
            # are internal to a kernel (e.g., triton kernel) such that these buffers
            # are never actually defined. This information is collected during codegen
            # and recorded in V.graph.removed_buffers. So we cleanup signature and write
            # prefix (i.e., generating call function and return outputs) after we have
            # codegen the partition.
            if not isinstance(V.graph.wrapper_code, SubgraphPythonWrapperCodegen):
                raise AssertionError(
                    "expected wrapper_code to be a SubgraphPythonWrapperCodegen"
                )
            removed_buffers_during_codegen = (
                V.graph.removed_buffers - removed_buffers_before_codegen
            )
            signature = self.clean_removed_buffer_from_partition_signatures(
                signature, removed_buffers_during_codegen
            )
            V.graph.wrapper_code.partition_signatures = signature
            V.graph.wrapper_code.write_prefix()

            partition_code, _ = V.graph.wrapper_code.generate(V.graph.is_inference)

        V.graph.wrapper_code.define_subgraph_launcher_fn(graph_name, partition_code)

        V.graph.wrapper_code.codegen_partition_call(graph_partition_id, signature)
        V.graph.wrapper_code.allocated.update(  # type: ignore[has-type]
            [node.get_name() for node in signature.output_nodes]
        )

    def use_default_device_context(
        self, partitions: list[PartitionType], signatures: list[GraphPartitionSignature]
    ) -> contextlib.AbstractContextManager[None]:
        # [FUNCTION DOC] Returns a context manager that decides on a shared default device context for all partitions and, if it needs a guard, enters the device guard on entry and exits it on exit (resetting the context).
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            # [FUNCTION DOC] HELPER: Context manager body implementing the default device guard enter/exit around partition codegen. Raises AssertionError if the device has no index.
            self.update_graph_partition_default_device(partitions, signatures)
            if self.default_device_context and device_need_guard(
                self.default_device_context.type
            ):
                if self.default_device_context.index is None:
                    raise AssertionError("device should have an index")
                V.graph.wrapper_code.codegen_device_guard_enter(
                    self.default_device_context.index
                )

            try:
                yield
            finally:
                if self.default_device_context and device_need_guard(
                    self.default_device_context.type
                ):
                    V.graph.wrapper_code.codegen_device_guard_exit()
                self.default_device_context = None

        return ctx()

    def update_graph_partition_default_device(
        self, partitions: list[PartitionType], signatures: list[GraphPartitionSignature]
    ) -> None:
        # [FUNCTION DOC] Sets default_device_context to the device of the first cudagraph partition when all non-cudagraph partitions run entirely on that device (to share the guard), except with a single cudagraph partition
        # or none; otherwise leaves it unset.
        # Note: [Graph Partition Device Contexts]
        # Entering a device context takes 60 microseconds and exiting a device
        # context takes 20 microseconds. If all graph partitions and
        # cudagraph-unsafe ops happen on the same device, we can share the
        # device context.

        if len(partitions) == 1 and not signatures[0].skip_cudagraph:
            # If there is only 1 cudagraph partition, the device context
            # should happen within the cudagraph partition, which
            # would be removed by cudagraph.
            return

        def get_cudagraph_partition_device(partition: PartitionType) -> torch.device:
            # [FUNCTION DOC] HELPER: Returns the device of the first node of a partition; raises AssertionError if missing.
            partition_device = partition[0].get_device()
            if partition_device is None:
                raise AssertionError("expected partition_device to be set")
            return partition_device

        def all_on_target_device(
            partition: PartitionType, target_device: torch.device
        ) -> bool:
            # [FUNCTION DOC] HELPER: True if every node of the partition is on target_device.
            for node in partition:
                device = node.get_device()
                if device != target_device:
                    return False
            return True

        cudagraph_partition_device = None
        for partition, signature in zip(partitions, signatures):
            if not signature.skip_cudagraph:
                cudagraph_partition_device = get_cudagraph_partition_device(partition)
                break

        # all partitions skip cudagraph
        if cudagraph_partition_device is None:
            return

        for partition, signature in zip(partitions, signatures):
            if signature.skip_cudagraph and not all_on_target_device(
                partition, cudagraph_partition_device
            ):
                return

        self.default_device_context = cudagraph_partition_device

    def _codegen_partitions(self) -> None:
        # [FUNCTION DOC] Splits the nodes into partitions and generates code for each: skipped partitions inline, cudagraph partitions as separate functions, within the default device context. Counts partitions in counters, sets
        # partition names on the wrapper and checks that the number of partition maps equals the number of generated partitions (AssertionError otherwise).
        """
        Split nodes into partitions and codegen each partition into separate functions.
        This allows further applying different optimizations (e.g., cudagraph) to
        each function.
        """
        partitions, signatures = self.graph_partition()

        if len(partitions) > 1:
            counters["inductor"]["cudagraph_partitions"] += len(partitions)

        with self.use_default_device_context(partitions, signatures):
            for partition, signature in zip(partitions, signatures):
                if len(partition) < 1:
                    raise AssertionError(
                        f"Each partition must have at least one node but found {len(partition)}"
                    )

                if signature.skip_cudagraph:
                    self._codegen(partition)
                else:
                    self._codegen_partition_wrapper(partition, signature)

        num_partitions = next(self._graph_partition_counter)
        V.graph.wrapper_code.set_all_partition_names(num_partitions)

        # See [Note: Graph Partition Map for CUDAGraph]
        if num_partitions > 0:
            if V.graph.partition_maps is None:
                raise AssertionError("expected V.graph.partition_maps to be set")
            if num_partitions != len(V.graph.partition_maps):
                raise AssertionError(
                    f"Expect {num_partitions} partition maps but got {len(V.graph.partition_maps)}"
                )

    def _codegen(self, nodes: list[BaseSchedulerNode]) -> None:
        # [FUNCTION DOC] HELPER: Generates code for the given nodes in order (all nodes or one partition). For each node: switches device guards (flushing and exiting streams as needed), enters mempool and stream contexts, emits
        # deferred alignment copies, records current node and buffers to free, then dispatches by kind (template, extern call, combo kernel, nested reduction, mix order reduction, fused/plain node via the device backend, or no-op
        # marking), adds debug sync, updates available buffers and completed operations, flushes when the backend is ready, and closes the mempool context. At the end exits the outer device guard when needed and flushes.
        # Optionally (testing config) checks that the Python stack has no repeated frames. Raises AssertionError on inconsistent state (previous node set, missing device index, unsupported node or backend type).
        # Side effects: writes wrapper code, mutates scheduler codegen state.
        if config.check_stack_no_cycles_TESTING_ONLY:
            import torch._dynamo.convert_frame

            stack = traceback.extract_stack()
            seen: OrderedSet[tuple[str, int | None]] = OrderedSet()
            for frame in reversed(stack):
                # This is where maybe_cprofile is
                if (
                    frame.name == "_compile_inner"
                    and frame.filename == torch._dynamo.convert_frame.__file__
                ):
                    break
                key = (frame.filename, frame.lineno)
                if key in seen:
                    raise AssertionError(
                        f"Duplicate stack frame {frame.filename}:{frame.lineno}; "
                        "did you add a decorator to one of the functions in this stack "
                        "trace?  If so, try using a context manager instead."
                    )
                seen.add(key)

        self.current_device = self.default_device_context
        if self.previous_node is not None:
            raise AssertionError("expected previous_node to be None")

        # pyrefly: ignore [unbound-name]
        if self.default_device_context and config.triton.autotune_at_compile_time:
            V.graph.wrapper_code.write_get_raw_stream_header()

        # Register non-mutated inputs that need alignment checks.
        # Deferred to just before the first kernel that reads each input.
        V.graph.wrapper_code.register_alignment_check_inputs()

        # An input read on more than one stream needs its copy_if_misaligned
        # emitted per consuming stream (not once at the first reader by line
        # order, which would place the only copy inside one branch's stream and
        # race the others).  Reclassify those inputs here.
        if self._has_multi_stream_nodes():
            pending = V.graph.wrapper_code._pending_alignment_copies
            if pending:
                input_streams: dict[str, OrderedSet[int]] = {}
                for n in nodes:
                    s = self.node_to_stream.get(n, 0)
                    for name in _real_dep_names(n.read_writes.reads):
                        if name in pending:
                            input_streams.setdefault(name, OrderedSet()).add(s)
                multi = [name for name, ss in input_streams.items() if len(ss) > 1]
                if multi:
                    V.graph.wrapper_code.mark_multistream_alignment(multi)

        for node in nodes:
            if log.isEnabledFor(logging.DEBUG):
                try:
                    log.debug(
                        "Generating code for node %s with estimated runtime %f",
                        node.get_name(),
                        node.get_estimated_runtime(),
                    )
                except Exception:
                    log.debug(
                        "Generating code for node %s with estimated runtime 0.0",
                        node.get_name(),
                    )

            self.enter_context(node)

            # pyrefly: ignore [unbound-name]
            if config.size_asserts:
                V.graph.wrapper_code.codegen_deferred_input_asserts(
                    dep.name for dep in node.read_writes.reads
                )

            if device := node.get_device():
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if self.current_device and device_need_guard(
                        self.current_device.type
                    ):
                        # Exit stream context before exiting device guard
                        if self.current_stream_idx is not None:
                            self.generate_stream_ctx_exit()
                        V.graph.wrapper_code.codegen_device_guard_exit()
                    self.current_device = device
                    if device_need_guard(device.type):
                        if device.index is None:
                            raise AssertionError("device should have an index")
                        # Compute num_streams if we have multi-stream nodes
                        num_streams = 1
                        if self._has_multi_stream_nodes():
                            # Count unique streams (excluding default stream 0)
                            unique_streams = OrderedSet(
                                self.get_node_stream(n) for n in self.nodes
                            )
                            num_streams = (
                                max(unique_streams) + 1 if unique_streams else 1
                            )
                        V.graph.wrapper_code.codegen_device_guard_enter(
                            device.index,
                            num_streams,
                            self.stream_idx_to_user_obj_idx,
                        )

            node_mempool = None
            mempool_ctx_entered = False
            # Nop nodes allocate nothing; avoid emitting an empty use_mem_pool
            # context even when their metadata has a pool.
            if not isinstance(node, NopKernelSchedulerNode):
                node_mempool = self.node_to_mempool.get(node)
            if node_mempool is not None:
                # Flush any previously queued wrapper code before entering a pool
                # that only applies to the current scheduler node. If a stream
                # context from the previous node is still open, close it first so
                # the generated order is use_mem_pool outer, stream inner.
                self.flush()
                if self.current_stream_idx is not None:
                    self.generate_stream_ctx_exit()
                V.graph.wrapper_code.codegen_cuda_mempool_enter(node_mempool)
                mempool_ctx_entered = True

            # Handle stream context switching for multi-stream scheduling.
            # This runs for all nodes (including device-less sync ops like
            # record_event/wait_event) so they are placed inside the correct
            # stream context. Only switch when inside a device guard (i.e.
            # current_device is set), since stream variables are declared there.
            if self._has_multi_stream_nodes() and self.current_device is not None:
                self.generate_stream_ctx_switching(node)

            # Emit deferred alignment copies for inputs first used by this
            # node, on this node's stream.  This runs *after* mempool and
            # stream context switching so the copy executes inside the same
            # pool and on the same stream as the consuming kernel; inputs read
            # on multiple streams get one copy per stream.
            V.graph.wrapper_code.codegen_deferred_alignment_copies(
                (dep.name for dep in node.read_writes.reads),
                self.node_to_stream.get(node, 0),
            )

            self.current_node = node
            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                prologue, template_node, epilogue = node.get_prologue_template_epilogue(
                    list(node.get_nodes())
                )
                # pyrefly: ignore [unbound-name]
                self.get_backend(device).codegen_template(
                    template_node, epilogue, prologue
                )
            elif node.is_extern():
                self.codegen_extern_call(node)
            elif node.is_foreach():
                node = typing.cast(ForeachKernelSchedulerNode, node)
                # pyrefly: ignore [unbound-name]
                backend_ = self.get_backend(device)
                from .codegen.cuda_combined_scheduling import CUDACombinedScheduling
                from .codegen.simd import SIMDScheduling
                from .codegen.xpu.xpu_combined_scheduling import XPUCombinedScheduling

                if isinstance(
                    backend_,
                    (SIMDScheduling, CUDACombinedScheduling, XPUCombinedScheduling),
                ):
                    backend = backend_
                else:
                    raise AssertionError(f"{type(self)=}")
                backend.codegen_combo_kernel(node)
            elif isinstance(node, FusedNestedReductions):
                # pyrefly: ignore [unbound-name]
                self.get_backend(device).codegen_nested_reduction(node)
            elif isinstance(node, FusedMixOrderReductions):
                # pyrefly: ignore [unbound-name]
                self.get_backend(device).codegen_mix_order_reduction(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                # pyrefly: ignore [unbound-name]
                self.get_backend(device).codegen_node(node)
            else:
                if not isinstance(node, NopKernelSchedulerNode):
                    raise AssertionError("expected node to be a NopKernelSchedulerNode")
                node.mark_run()

            # pyrefly: ignore [unbound-name]
            if config.triton.debug_sync_kernel:
                # pyrefly: ignore [unbound-name]
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_buffer_names())
            self.completed_operations.update(node.get_operation_names())

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if (
                    device is not None
                    and device.type != "meta"
                    and self.get_backend(device).ready_to_flush()
                ):
                    self.flush()

            if mempool_ctx_entered:
                # Ensure backends that emit wrapper code at flush time still
                # allocate and launch inside the active pool and stream.
                self.flush()
                if self.current_stream_idx is not None:
                    self.generate_stream_ctx_exit()
                V.graph.wrapper_code.codegen_cuda_mempool_exit()

            if all(isinstance(n, SchedulerNode) for n in node.get_nodes()):
                self.previous_node = node
            else:
                self.previous_node = None

        if self.current_device != self.default_device_context:
            # when default_device_context is not None, we are codegen
            # for graph partitions and all nodes must be on
            # the same default device.
            if self.current_device is None:
                raise AssertionError("expected current_device to be set")
            if device_need_guard(self.current_device.type):
                # exit the outermost CUDA device guard. this is
                # important for nested indentation codegen-ing.
                V.graph.wrapper_code.codegen_device_guard_exit()

        self.previous_node = None
        self.flush()

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode], node_benchmark_results
    ) -> tuple[float, float, list[str | None]]:
        # [FUNCTION DOC] Benchmarks a combo kernel built from node_list on the first node's device through the backend, returning (combo time ms, clone/copy time ms, kernel paths). Sets V.graph.scheduler and current_device.
        # Raises AssertionError if the device is missing.
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        device = node_list[0].get_device()
        V.graph.scheduler = self
        self.current_device = device
        if device is None:
            raise AssertionError("expected device to be set")
        backend = self.get_backend(device)
        return backend.benchmark_combo_kernel(node_list, node_benchmark_results)

    def speedup_by_combo_kernel(self, nodes: list[BaseSchedulerNode]) -> bool:
        # [FUNCTION DOC] Returns whether combining the nodes into one combo kernel is profitable. True without benchmarking when combo kernel benchmarking is off. Otherwise benchmarks each node separately (any register spilling
        # rejects; a loop-carried variable compile error allows) and the combined kernel, and accepts when the combined time minus clone time beats the sum or the kernels are very small (under 0.3 ms). Raises AssertionError
        # if nodes are on different devices.
        """
        If config.benchmark_fusion is False, always return True.
        Otherwise, return True if fusion can brings speedup.
        """

        subkernel_nodes = nodes
        device = subkernel_nodes[0].get_device()

        if not all(node.get_device() == device for node in subkernel_nodes):
            raise AssertionError(
                "All nodes in a combo kernel group must be on the same device"
            )

        if not config.benchmark_combo_kernel:
            return True

        from triton.compiler.errors import CompilationError

        ms1, path1_list = 0.0, []
        node_benchmark_results = {}
        for i, snode in enumerate(subkernel_nodes):
            node_list = snode.get_nodes()
            # We can not accurately benchmark kernel using atomic_add
            # due to how we generate random integer inputs.
            if self._any_atomic_add(node_list):
                fusion_log.debug(
                    "ComboKernel: benchmarking may not accurate due to atomic_add"
                )

            try:
                ms, path = self.benchmark_fused_nodes(node_list)
                node_benchmark_results[snode] = (ms, path)
                if math.isinf(ms):
                    fusion_log.debug(
                        "ComboKernel benchmark: register spilling of %d-th subkernel",
                        i,
                    )
                    return False
            except CompilationError as e:
                # workaround triton issue: https://github.com/triton-lang/triton/issues/2151
                if "Loop-carried variable" in str(e):
                    fusion_log.debug(
                        "ComboKernel benchmark: return True because of loop-carried variable"
                    )
                    return True  # allow fusion
                else:
                    raise
            ms1 += ms
            path1_list.append(path)

        try:
            ms2, ms2_clone, _path2_list = self.benchmark_combo_kernel(
                subkernel_nodes, node_benchmark_results
            )
        except CompilationError as e:
            # workaround triton issue: https://github.com/triton-lang/triton/issues/2151
            if "Loop-carried variable" in str(e):
                fusion_log.debug(
                    "ComboKernel benchmark: return True because of loop-carried variable"
                )
                return True  # allow fusion
            else:
                raise

        # small kernels are very likely to have speedup but hard to benchmark. So we skip benchmarking.
        small_kernel = ms2 - ms2_clone < 0.3 or ms1 < 0.3
        if fusion_log.isEnabledFor(logging.DEBUG):
            if ms1 > ms2 or small_kernel:
                fusion_log.debug(
                    "can fuse (benchmark): fusing causes %sx speedup",
                    green_text(f"{ms1 / ms2:.3f}"),
                )
            else:
                fusion_log.debug(
                    "cannot fuse (benchmark): fusing causes %sx slowdown",
                    red_text(f"{ms1 / ms2:.3f}"),
                )
        # ms1 returned by benchmark_fused_nodes discounted clone time
        return ms2 - ms2_clone < ms1 or small_kernel

    def get_buffer_layout(self, buf_name: str) -> ir.Layout:
        # [FUNCTION DOC] Returns the layout of the named buffer; raises AssertionError if the buffer has no IR node.
        buf = self.name_to_buf[buf_name]
        if buf.node is None:
            raise AssertionError("expected buf.node to be set")
        return buf.node.get_layout()

    def update_zero_dim_cpu_tensor(self) -> None:
        # [FUNCTION DOC] Records in V.graph.zero_dim_cpu_tensor_list the names of zero-dimensional CPU buffers read by GPU nodes (mutates graph state).
        for node in self.nodes:
            if node.is_gpu():
                for read in node.read_writes.reads:
                    buffer = V.graph.name_to_buffer.get(read.name)
                    if (
                        buffer
                        and get_device_type(buffer) == "cpu"
                        and not isinstance(
                            buffer.layout, (NoneLayout, MultiOutputLayout)
                        )
                        and buffer.get_size() == []
                    ):
                        V.graph.zero_dim_cpu_tensor_list.add(read.name)

    @property
    def current_stream_idx(self) -> int | None:
        # [FUNCTION DOC] Property: stream index of the stream context currently entered in codegen, or None if none.
        """CUDA Stream index that current scheduler node assigned to."""
        if self._current_stream_ctx is not None:
            return self._current_stream_ctx.stream_idx
        else:
            return None

    @property
    def current_stream_name(self) -> str | None:
        # [FUNCTION DOC] Property: stream name of the current stream context, or None if none.
        """CUDA Stream name that current scheduler node assigned to."""
        if (stream_idx := self.current_stream_idx) is not None:
            return get_stream_name(stream_idx)
        else:
            return None

    def generate_stream_ctx_enter(self, node: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] Emits code to enter the stream assigned to the node and records the resulting context. Raises AssertionError for no-op nodes.
        """Code-gen to enter the Stream context assigned to node."""
        if isinstance(node, NopKernelSchedulerNode):
            raise AssertionError("expected node to not be a NopKernelSchedulerNode")
        node_stream = self.get_node_stream(node)
        self._current_stream_ctx = V.graph.wrapper_code.codegen_cuda_stream_enter(
            stream_idx=node_stream,
        )

    def generate_stream_ctx_exit(self) -> None:
        # [FUNCTION DOC] Emits code to exit the current stream context and clears it; raises AssertionError if none is active.
        """Code-gen to exit from the current Stream context."""
        if self._current_stream_ctx is None:
            raise AssertionError("expected _current_stream_ctx to be set")
        V.graph.wrapper_code.codegen_cuda_stream_exit()
        self._current_stream_ctx = None

    def generate_stream_ctx_switching(self, node: BaseSchedulerNode) -> None:
        # [FUNCTION DOC] Emits stream enter/exit code so the node runs on its assigned stream: nothing if the stream is unchanged or the node is a no-op inside an existing context, enters when no context is active, otherwise
        # exits the old and enters the new context.
        """Generate stream entering and exiting to properly run node in a multi-stream scenario.

        Stream context switching is only generated if ``node``'s assigned stream is different from
        the previous node's stream. NopKernelSchedulerNodes have stream=None and inherit the
        enclosing stream context (or do nothing if no context is active yet).
        """
        stream = (
            None
            if isinstance(node, NopKernelSchedulerNode)
            else self.get_node_stream(node)
        )
        if self.current_stream_idx == stream:
            # Covers: same stream as current (no switch needed), and both None
            # (nop node before any stream context — nothing to do).
            return
        elif self.current_stream_idx is not None and stream is None:
            # Don't generate ctx switching. Memory planning code (e.g., delete buffers) on current
            # node goes to previous stream ctx.
            return
        elif self.current_stream_idx is None and stream is not None:
            # Enter new ctx, update current stream status.
            self.generate_stream_ctx_enter(node)
        else:
            # Switching from previous stream ctx to the new stream ctx.
            self.generate_stream_ctx_exit()
            self.generate_stream_ctx_enter(node)


class BaseScheduling:  # noqa: docstring_linter
    # [CLASS DOC] Base class (interface) for device scheduling backends used by Scheduler (holds the scheduler reference). Defines fusion legality hooks (vertical, horizontal, reduction epilogue, multi-output template), fusing of
    # two nodes (dispatching to the fused node classes), grouping of iteration sizes, code generation hooks (template, node, mix order, nested reduction, sync, flush), benchmarking hooks and fusion priority. Subclasses
    # implement the abstract methods, which raise NotImplementedError here.
    def __init__(self, scheduler: Scheduler | None):
        # [FUNCTION DOC] Stores the owning scheduler (may be None).
        super().__init__()
        self.scheduler = scheduler

    def free_buffers_in_scheduler(self) -> None:
        # [FUNCTION DOC] Asks the scheduler, if present, to free buffers pending release.
        if self.scheduler:
            self.scheduler.free_buffers()

    def get_backend_features(self, device: torch.device) -> OrderedSet[BackendFeature]:
        # [FUNCTION DOC] Returns the set of BackendFeature supported on the device; the base returns an empty set.
        """Return a set of .codegen.common.BackendFeature()"""
        return OrderedSet()

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Abstract: whether node1 and node2 can be vertically fused on this backend; raises NotImplementedError.
        """
        Check whether node1 and node2 can be vertically fused or not.
        """
        raise NotImplementedError

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Abstract: whether node1 and node2 can be horizontally fused on this backend; raises NotImplementedError.
        """
        Check whether node1 and node2 can be horizontally fused or not.
        """
        raise NotImplementedError

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] Whether a reduction epilogue can be fused into a template; base returns False.
        return False

    def can_fuse_multi_outputs_template(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # [FUNCTION DOC] True if node1 is a multi-output template and node2 is a MultiOutput node taking exactly that template buffer as its only input; False otherwise.
        """
        A Multi-Output Template (referenced in #144012) is a template node
        with MultiOutputLayout, and its output buffers are instances of MultiOutput.
        In this context, we verify whether node1 represents the Multi-Output Template
        and node2 corresponds to one of its outputs. If so, we further check if
        backend supports this fusion.

        """
        template_buf = node1.get_template_node()
        if not isinstance(template_buf, ir.TemplateBuffer):
            return False
        if not template_buf.is_multi_outputs_template():
            return False

        if isinstance(node2.node, ir.MultiOutput):
            return (
                len(node2.node.inputs) == 1
                and isinstance(node2.node.inputs[0], ir.IRNode)
                and node2.node.inputs[0].get_name() == template_buf.get_name()
            )

        return False

    def fuse(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> FusedSchedulerNode:
        # [FUNCTION DOC] Fuses two nodes into a fused node, dispatching: foreach fusion if either is foreach; nested reduction fusion for fusable dependent reductions; mix order reduction fusion; appending into an existing nested or
        # mix-order node; epilogue fusion of a user Triton extern kernel with a scheduler node (raises AssertionError if the extern node is not a user Triton kernel); otherwise plain fused node.
        """
        Fuse two nodes
        """
        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        elif NestedReduction._is_dependent_reduction_pair(
            node1, node2
        ) and NestedReduction.can_fuse(node1, node2):
            return FusedNestedReductions(node1, node2)
        elif MixOrderReduction.are_mix_order_reductions(node1, node2):
            return FusedMixOrderReductions(node1, node2)
        elif isinstance(node1, FusedNestedReductions):
            return node1.fuse_with(node2)
        elif isinstance(node1, FusedMixOrderReductions):
            return node1.fuse_with(node2)
        elif isinstance(node1, ExternKernelSchedulerNode) and isinstance(
            node2, SchedulerNode
        ):
            if not isinstance(node1.node, ir.UserDefinedTritonKernel):
                raise AssertionError(
                    "expected node1.node to be an ir.UserDefinedTritonKernel"
                )
            return FusedExternTritonKernelSchedulerNode.epilogue_fuse(node1, node2)
        else:
            return FusedSchedulerNode.fuse(node1, node2)

    def group_fn(
        self, sizes: Sequence[Sequence[sympy.Expr]]
    ) -> tuple[tuple[sympy.Expr, ...], ...]:
        # [FUNCTION DOC] Abstract: transforms iteration sizes if the backend needs it; raises NotImplementedError.
        """
        Process the iteration sizes in case a transformation needs to be applied.
        """
        raise NotImplementedError

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> str | None:
        # [FUNCTION DOC] Abstract: generates a kernel for a template node with epilogue and prologue nodes (Triton only today); raises NotImplementedError.
        """
        Given a template node, generate a kernel.

        This function is only available for triton now. If the third-party backend behaves as a sub-class
        of TritonScheduling, it can override it or reuse it.
        """
        raise NotImplementedError

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[BaseSchedulerNode],
        benchmark_kernel: bool,
        hint_override: int | None = None,
    ) -> str:
        # [FUNCTION DOC] Abstract: generates kernel source code for pre-fused nodes (optionally for benchmarking and with a hint override); raises NotImplementedError.
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        raise NotImplementedError

    def codegen_node(self, node: FusedSchedulerNode | SchedulerNode) -> None:
        # [FUNCTION DOC] Abstract: generates a kernel for a fused or plain scheduler node; raises NotImplementedError.
        """
        Generate a kernel given a list of pre-fused nodes.
        """
        raise NotImplementedError

    def codegen_mix_order_reduction(self, node: FusedMixOrderReductions) -> None:
        # [FUNCTION DOC] Abstract: generates code for a mix order reduction node; raises NotImplementedError.
        raise NotImplementedError

    def codegen_nested_reduction(self, node: FusedNestedReductions) -> None:
        # [FUNCTION DOC] Abstract: generates code for a nested reduction node; raises NotImplementedError.
        raise NotImplementedError

    def codegen_sync(self) -> None:
        # [FUNCTION DOC] Abstract: generates hardware dependent synchronization code; raises NotImplementedError.
        """
        Generate synchronization code for the kernel. This method depends on the hardware characteristics.
        """
        raise NotImplementedError

    def ready_to_flush(self) -> bool:
        # [FUNCTION DOC] Whether the backend asks the scheduler to flush generated kernels now; base returns False.
        """
        Check whether the backend is requesting the scheduler to flush the generated kernel.
        If not supported, please return False.
        """
        return False

    def flush(self) -> None:
        # [FUNCTION DOC] Abstract: flushes generated kernel and wrapper code to the source file; raises NotImplementedError.
        """
        Flush the generated kernel and python wrapper code to the source code file.
        """
        raise NotImplementedError

    def benchmark_fused_nodes(
        self, nodes: Sequence[BaseSchedulerNode]
    ) -> tuple[float, str]:
        # [FUNCTION DOC] Abstract: benchmarks fused nodes on random inputs and returns (milliseconds, path); raises NotImplementedError.
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError

    def benchmark_codegened_module(self, module: ModuleType) -> tuple[float, str]:
        # [FUNCTION DOC] Abstract: benchmarks a compiled module and returns (milliseconds, path); raises NotImplementedError.
        """
        Benchmark a compiled module and return the execution time
        in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError

    def get_fusion_pair_priority(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> int:
        # [FUNCTION DOC] Returns the fusion pair priority as an unsigned integer (smaller is higher priority); base returns 0.
        """
        Return an unsigned integer which represents the priority of this fusion pair.
        The smaller is with higher priority.
        """
        return 0

    def benchmark_combo_kernel(
        self, node_list: Sequence[BaseSchedulerNode], node_benchmark_results
    ) -> tuple[float, float, list[str | None]]:
        # [FUNCTION DOC] Abstract: benchmarks nodes combined in a combo kernel, returning combined and copy times and paths; raises NotImplementedError.
        """
        Benchmark the list of nodes to combine and return the execution time
        and memory copy time in milliseconds on randomly generated inputs.
        """
        raise NotImplementedError

    def codegen_comment(
        self,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel_name: str | None = None,
    ) -> None:
        # [FUNCTION DOC] When a kernel name is given, records post-grad provenance tracing for the node schedule and writes the debug handle into the wrapper code (side effect).
        if kernel_name:
            from torch._inductor.debug import set_kernel_post_grad_provenance_tracing

            debug_handle = set_kernel_post_grad_provenance_tracing(
                node_schedule,  # type: ignore[arg-type]
                kernel_name,
            )
            V.graph.wrapper_code.write_provenance_debug_handle(
                kernel_name, debug_handle
            )
