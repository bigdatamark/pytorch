# [MODULE DOC] Runtime estimation helpers for the overlap scheduler: collective cache keys, NCCL and Inductor analytical
# estimates, CUDA-event benchmarking of collectives with a process-local cache, and structured-trace (tlparse) CSV artifact logging
# of compute and collective estimations. Import-time side effect: creates the node_runtime_estimation artifact logger.
"""
Node runtime estimation for overlap scheduling.
"""

from __future__ import annotations

import functools
import itertools
import operator
from functools import lru_cache
from typing import Any

import torch
import torch.fx as fx
from torch._inductor import config
from torch._inductor.fx_passes.bucketing import (
    _resolve_group_name,
    _schedulable_wait_node,
)
from torch._inductor.utils import clear_on_fresh_cache
from torch._logging import getArtifactLogger, trace_structured
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.operator_schemas import normalize_function


def _format_csv(headers: list[str], rows: list[list[str]]) -> str:
    # [FUNCTION DOC] HELPER: Joins headers and rows into a CSV string (comma separated, newline between lines, values converted to text).
    """Format data as CSV. Human-readable and easily parseable."""
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines)


def _get_collective_key(coll_node: fx.Node) -> str:
    # [FUNCTION DOC] HELPER: Builds a string key for a collective fx node from its target, group size, resolved group name and the byte size
    # of the first input tensor found in its fake args (None if unavailable or any dimension has no hint).
    # Raises an error if the node arguments cannot be normalized.
    """Extract a unique key for a collective node including group info and tensor size."""
    from torch._inductor import fx_utils

    opt_args_kwargs = normalize_function(
        coll_node.target,  # type: ignore[arg-type]
        args=coll_node.args,
        kwargs=coll_node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt_args_kwargs is None:
        raise AssertionError("normalize_function returned None for collective node")
    _, kwargs = opt_args_kwargs
    raw_group_name = kwargs.get("group_name", None)
    group_name = (
        _resolve_group_name(raw_group_name) if raw_group_name is not None else None
    )
    group_size = kwargs.get("group_size", None)

    tensor_bytes: int | None = None
    success, args, kw = fx_utils.get_fake_args_kwargs(coll_node)
    if success:

        def extract_first_tensor_bytes(t: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Tree-map callback that, for the first tensor seen, records its byte size (product of hinted dims times element size)
            # into the enclosing variable; leaves it unset if any dimension lacks a hint. Returns the tensor unchanged.
            nonlocal tensor_bytes
            if tensor_bytes is None:
                shape = [get_hint(dim) for dim in t.shape]
                if all(s is not None for s in shape):
                    numel = functools.reduce(operator.mul, shape, 1)
                    tensor_bytes = numel * t.dtype.itemsize
            return t

        torch.utils._pytree.tree_map_only(
            torch.Tensor, extract_first_tensor_bytes, (args, kw)
        )

    return f"{coll_node.target} group_size:{group_size} group_name:{group_name} input_bytes:{tensor_bytes}"


def _get_collective_estimations(coll_node: fx.Node) -> tuple[float, float]:
    # [FUNCTION DOC] HELPER: Returns the pair (nccl_ms, inductor_ms) of estimated collective runtimes in milliseconds for an fx node,
    # computed once with the NCCL estimator and once with Inductor's analytical model.
    """Get NCCL and Inductor analytical estimations for a collective node.

    Returns: (nccl_ms, inductor_ms)
    """
    nccl_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=True
        )
    )
    inductor_ms = (
        torch._inductor.comm_analysis.estimate_nccl_collective_runtime_from_fx_node(
            coll_node, None, use_nccl_estimator=False
        )
    )
    return nccl_ms, inductor_ms


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")


# TODO: Consider using a distributed-aware cache or rank-local disk cache
# not using local cache because different ranks might write to it concurrently.
# solvable in future, potentially with workflow to seed cache
@clear_on_fresh_cache
@lru_cache
def _get_collective_cache() -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Returns the process-local mapping from collective key to benchmarked runtime. Cached per process via a memoized
    # call and cleared when the Inductor fresh-cache hook fires.
    """Get process-local cache for collective benchmarks."""
    return {}


def get_cached_runtime(key: str) -> float | None:
    # [FUNCTION DOC] Returns the cached benchmarked runtime for the key, or None if absent.
    """Get cached runtime from process-local cache."""
    return _get_collective_cache().get(key)


def set_cached_runtime(key: str, value: float) -> None:
    # [FUNCTION DOC] Stores a benchmarked runtime for the key in the process-local cache. Mutates the cache.
    """Set cached runtime in process-local cache."""
    _get_collective_cache()[key] = value


def get_hint(x: int | torch.SymInt) -> int | None:
    # [FUNCTION DOC] Returns the concrete integer for an int or a SymInt with a hint, or None for a SymInt without a hint.
    # Raises an error if the argument is neither int nor SymInt.
    if isinstance(x, int):
        return x
    if not isinstance(x, torch.SymInt):
        raise AssertionError(f"expected int or SymInt, got {type(x)}")
    return x.hint if x.has_hint() else None


def can_benchmark_collective() -> bool:
    # [FUNCTION DOC] Returns True only if torch.distributed is initialized and the default group's backend is not the fake backend,
    # i.e. real collectives can be run for benchmarking.
    """Check if we can benchmark collectives (not fake process group)."""
    import torch.distributed as c10d

    if not c10d.is_initialized():
        return False

    pg = c10d.distributed_c10d._get_default_group()
    if (
        torch.distributed.distributed_c10d.get_backend(pg)
        == torch.distributed.distributed_c10d.Backend.FAKE
    ):
        return False

    return True


def _median(lst):
    # [FUNCTION DOC] HELPER: Returns the median of a non-empty list of numbers as a Python float; raises an error on an empty list.
    if len(lst) == 0:
        raise AssertionError("expected non-empty list for median")
    return torch.median(torch.tensor(lst)).item()


def _benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    nruns: int,
) -> float | None:
    # [FUNCTION DOC] HELPER: Benchmarks one collective node. Converts fake tensor args to real random strided tensors using hinted shapes/strides,
    # replaces SymInts by optimization hints (with the configured unbacked fallback), runs one warmup call with synchronization, then
    # nruns timed calls (each followed by a wait on the result) measured with timing events. Returns the median time in ms.
    # Raises an error if a tensor has a dimension with no hint. Side effects: executes real collectives; requires an accelerator.
    """
    Core benchmarking logic using CUDA events and barriers.
    Returns runtime in ms or None on failure.
    """
    from torch._dynamo.testing import rand_strided

    # Convert FakeTensors to real tensors before benchmarking
    def to_real(t: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Converts a fake tensor into a real random tensor with the same hinted shape, stride, device and dtype.
        # Raises an error if any size or stride has no hint.
        shape = [get_hint(dim) for dim in t.shape]
        stride = [get_hint(s) for s in t.stride()]

        if any(s is None for s in itertools.chain(shape, stride)):
            # This should not happen, as can_benchmark_collective checks for unbacked
            raise ValueError("Cannot convert tensor with symbolic dimensions")

        return rand_strided(shape, stride, device=t.device, dtype=t.dtype)  # type: ignore[arg-type]

    args, kwargs = torch.utils._pytree.tree_map_only(
        torch.Tensor,
        to_real,
        (args, kwargs),
    )

    args, kwargs = torch.utils._pytree.tree_map_only(
        torch.SymInt,
        lambda s: optimization_hint(s, fallback=config.unbacked_symint_fallback),
        (args, kwargs),
    )

    # Warmup: call collective once and wait
    torch.accelerator.synchronize()
    result = n.target(*args, **kwargs)  # type: ignore[operator]
    torch.ops._c10d_functional.wait_tensor(result)
    torch.accelerator.synchronize()

    # Benchmark with CUDA events
    comm_times = []
    for _ in range(nruns):
        start_evt = torch.Event(enable_timing=True)
        end_evt = torch.Event(enable_timing=True)

        start_evt.record()
        result = n.target(*args, **kwargs)  # type: ignore[operator]
        torch.ops._c10d_functional.wait_tensor(result)
        end_evt.record()
        end_evt.synchronize()

        comm_times.append(start_evt.elapsed_time(end_evt))

    return _median(comm_times)


def benchmark_collective_with_cuda_events(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    # [FUNCTION DOC] Public wrapper that benchmarks a collective node with timing events while all current dispatch modes are disabled
    # (needed because a context manager is not allowed under the profiler). Returns (runtime_ms, cache_key), or (None, empty string)
    # on failure. Default nruns is 2.
    # KNOWN ISSUE: the docstring describes the same behavior as the impl function; the two have different default nruns (2 vs 3).
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    # context manager not allowed with profiler.
    with torch.utils._python_dispatch._disable_current_modes():
        return benchmark_collective_with_cuda_events_impl(n, nruns)


def benchmark_collective_with_cuda_events_impl(
    n: torch.fx.Node,
    nruns: int = 3,
) -> tuple[float | None, str]:
    # [FUNCTION DOC] Computes a cache key from target, group size and input size in bytes, returns the cached runtime if present,
    # otherwise benchmarks via the private helper and caches the result. Returns (runtime_ms, key); (None, empty string) if collectives
    # cannot be benchmarked, fake args are unavailable, or no tensor size is known; (None, key) if the benchmark returns None.
    # Raises an error if the node arguments cannot be normalized. Side effect: updates the process-local cache.
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.
    """
    from torch._inductor import fx_utils
    from torch.distributed.distributed_c10d import _get_group_size_by_name

    # Early check: can we actually run collectives?
    if not can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(n)

    opt_args_kwargs = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt_args_kwargs is None:
        raise AssertionError("normalize_function returned None for collective node")
    group_name = _resolve_group_name(opt_args_kwargs[1]["group_name"])
    group_size = _get_group_size_by_name(group_name)

    if not success:
        return None, ""

    # Extract actual input size in BYTES (first tensor argument)
    actual_bytes: int | None = None

    def extract_tensor_info(t: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Tree-map callback that records, for the first tensor with fully hinted shape, its size in bytes in the enclosing
    # variable; later tensors (e.g. out-variants) are ignored. Returns the tensor unchanged.
        nonlocal actual_bytes
        if actual_bytes is None:
            shape = [get_hint(dim) for dim in t.shape]
            if any(s is None for s in shape):
                return t

            total_elems = 1
            for dim in shape:
                if dim is None:
                    raise AssertionError(f"expected non-None dim, got {dim}")
                total_elems *= dim

            actual_bytes = total_elems * t.dtype.itemsize
        else:
            # out-variants (e.g. all_gather_into_tensor_out) can have multiple tensors
            pass
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, extract_tensor_info, (args, kwargs))

    if actual_bytes is None:
        return None, ""

    # Cache key by BYTES (dtype-agnostic)
    key = f"{n.target}: ({group_size} group size, {actual_bytes} bytes)"

    # Check cache
    if (cached := get_cached_runtime(key)) is not None:
        return cached, key

    # Benchmark using CUDA events with actual args/kwargs
    runtime = _benchmark_collective_with_cuda_events_impl(n, args, kwargs, nruns)

    if runtime is None:
        return None, key

    # Cache the result
    set_cached_runtime(key, runtime)
    return runtime, key


def _log_compute_estimations(
    compute_nodes: list[fx.Node],
    benchmarked_estimations: list[float],
    analytical_estimations: list[float],
) -> None:
    # [FUNCTION DOC] HELPER: Emits a structured-trace artifact with a CSV comparing benchmarked and analytical runtime estimates per compute node
    # (summary with dtypes and shapes of node inputs, both estimates converted to microseconds, ratio, difference and flop count).
    # Inputs are three parallel lists. Side effect: writes the trace artifact. Returns nothing.
    """Log compute node runtime estimations comparing benchmarked vs analytical."""
    import torch.utils._pytree as pytree
    from torch._inductor.fx_utils import count_flops_fx
    from torch.utils._dtype_abbrs import dtype_abbrs

    def _node_summary(n: fx.Node) -> str:
        # [FUNCTION DOC] HELPER: Returns the node name followed by dtype abbreviation and shape of each fx-node argument with tensor metadata.
        ret = str(n)
        for arg in pytree.arg_tree_leaves(n.args, n.kwargs):
            if not isinstance(arg, torch.fx.Node):
                continue
            if "val" in arg.meta:
                t = arg.meta["val"]
                if isinstance(t, torch.Tensor):
                    ret += f" {dtype_abbrs[t.dtype]}{tuple(t.shape)}"
                elif isinstance(t, torch.SymInt):
                    ret += f" SymInt({t})"
        return ret

    headers = [
        "Node",
        "Benchmarked Est(us)",
        "Analytical Est(us)",
        "Diff(ratio)",
        "Diff(us)",
        "Flops",
    ]

    def _fmt(v: object) -> str:
        # [FUNCTION DOC] HELPER: Formats a number to four decimals; symbolic values are shown with their hint when one exists.
        if isinstance(v, torch.types.py_sym_types):
            if v.node.has_hint():
                return f"{v}({v.node.hint:.4f})"
            return str(v)
        return f"{v:.4f}"

    rows = [
        [
            _node_summary(node),
            _fmt(est_b * 1e3),
            _fmt(est_a * 1e3),
            _fmt((est_a / est_b) if est_b > 0 else 0),
            _fmt((est_a - est_b) * 1e3),
            str(count_flops_fx(node)),
        ]
        for node, est_b, est_a in zip(
            compute_nodes, benchmarked_estimations, analytical_estimations
        )
    ]

    log_str = _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_compute_nodes_runtime_estimation",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )


def _log_graph_collective_benchmarks(gm: fx.GraphModule, artifact_name: str) -> None:
    # [FUNCTION DOC] HELPER: Walks the graph, finds each schedulable wait node, takes its start collective, benchmarks it (5 runs; failures count as 0.0),
    # and logs all results with world size under the given artifact name. Does nothing if no collectives are found.
    collective_nodes = []
    collective_keys = []
    benchmarked = []

    for node in gm.graph.nodes:
        if _schedulable_wait_node(node):
            start = node.args[0]
            collective_nodes.append(start)
            collective_keys.append(_get_collective_key(start))
            benchmarked_ms, _ = benchmark_collective_with_cuda_events(start, nruns=5)
            benchmarked.append(benchmarked_ms if benchmarked_ms else 0.0)

    if collective_nodes:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        _log_collective_benchmarks(
            collective_nodes,
            collective_keys,
            benchmarked,
            world_size,
            artifact_name,
        )


def _log_collective_benchmarks(
    collective_nodes: list[fx.Node],
    collective_keys: list[str] | None = None,
    benchmarked_medians: list[float] | None = None,
    world_size: int | None = None,
    artifact_name: str = "fx_collectives_analytical_estimation",
) -> None:
    # [FUNCTION DOC] HELPER: Logs a CSV artifact for tlparse comparing NCCL and Inductor estimates of each collective, and benchmarked medians
    # with ratios of estimate to benchmark if provided (ratio is 0 when the benchmark is not positive). Keys are computed if not given;
    # world size defaults to the distributed world size or 1. Side effect: writes the structured trace. Returns nothing.
    """Log collective estimations for tlparse. Includes benchmarks if provided."""
    if world_size is None:
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

    has_benchmarks = benchmarked_medians is not None

    if has_benchmarks:
        headers = [
            "Collective Key",
            "Benchmarked(ms)",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
            "NCCL Diff(ratio)",
            "Inductor Diff(ratio)",
        ]
    else:
        headers = [
            "Collective Key",
            "NCCL Est(ms)",
            "Inductor Est(ms)",
        ]

    rows = []
    for i, coll_node in enumerate(collective_nodes):
        key = collective_keys[i] if collective_keys else _get_collective_key(coll_node)
        nccl_ms, inductor_ms = _get_collective_estimations(coll_node)

        if benchmarked_medians is not None:
            benchmarked_ms = benchmarked_medians[i]
            nccl_diff_pct = (nccl_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            inductor_diff_pct = (
                (inductor_ms / benchmarked_ms) if benchmarked_ms > 0 else 0
            )
            rows.append(
                [
                    key,
                    f"{benchmarked_ms:.4f}",
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                    f"{nccl_diff_pct:.2f}",
                    f"{inductor_diff_pct:.2f}",
                ]
            )
        else:
            rows.append(
                [
                    key,
                    f"{nccl_ms:.4f}",
                    f"{inductor_ms:.4f}",
                ]
            )

    log_str = f"# World size: {world_size}\n"
    log_str += _format_csv(headers, rows)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )
