# [MODULE DOC] Graph pass that swaps FSDP all-gather / reduce-scatter collectives for symmetric-memory
# copy-engine "low contention" variants, to reduce SM and NVLink contention with overlapping compute.
# Main entry: replace_collectives_with_low_contention; the rest are helpers that classify collectives,
# filter by size and overlap, and locate the matching wait node.
# Import-time side effects: only creates a module logger.
from __future__ import annotations

import logging
import warnings

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _get_collective_info(node):
    # [FUNCTION DOC] HELPER: Classify a node as an all-gather or reduce-scatter functional collective.
    # Returns a pair (is_all_gather, group_name) or None if the node is neither. Imports bucketing and
    # overlap_scheduling helpers lazily.
    """Return (is_ag, group_name) if node is an AG/RS collective, else None."""
    from torch._inductor.fx_passes.bucketing import (
        is_all_gather_into_tensor,
        is_reduce_scatter_tensor,
    )
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name

    if is_all_gather_into_tensor(node):
        return True, get_group_name(node)
    if is_reduce_scatter_tensor(node):
        return False, get_group_name(node)
    return None


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
) -> None:
    # [FUNCTION DOC] Rewrite, in place, eligible all-gather and reduce-scatter nodes of the graph to symm_mem
    # low-contention ops. Collects collectives, drops those whose process group cannot be enabled for symmetric
    # memory, then skips ones that are too small (per-rank bytes below the configured minimum), lack compute-bound
    # overlap, or overlap collectives of other groups (the last two checks are bypassed by the skip-overlap config).
    # For all-gather the target op is chosen (optionally the multicast copy-engine variant); a node is skipped if
    # no out-variant replacement exists. Mutates the graph (replaces and erases nodes); returns nothing; logs counts.
    """Replace FSDP collectives with copy-engine symm_mem variants."""
    symm_mem = torch.ops.symm_mem

    collectives = []
    groups: OrderedSet[str] = OrderedSet()
    for node in list(graph.nodes):
        info = _get_collective_info(node)
        if info is None:
            continue
        is_ag, group_name = info
        collectives.append((node, is_ag, group_name))
        groups.add(group_name)

    if not collectives:
        return

    # Some group names can't be resolved at compile time — skip them.
    valid_groups: OrderedSet[str] = OrderedSet()
    for group_name in groups:
        if _enable_symm_mem(group_name):
            valid_groups.add(group_name)

    # Filter to collectives whose groups we can actually resolve
    collectives = [
        (node, is_ag, gn) for node, is_ag, gn in collectives if gn in valid_groups
    ]
    if not collectives:
        return

    from torch._inductor import config

    cfg = config.aten_distributed_optimizations
    min_bytes = cfg.low_contention_min_bytes_per_rank
    skip_overlap_check = cfg.low_contention_skip_overlap_check
    use_ag_ce_multicast = cfg.low_contention_all_gather_ce_multicast

    replacements = 0
    skipped_small = 0
    skipped_no_overlap = 0
    skipped_nvlink_contention = 0
    skipped_out_variant = 0
    for node, is_ag, group_name in collectives:
        coll_type = "AG" if is_ag else "RS"

        # Size filter: LC barrier overhead dominates for small messages
        if min_bytes > 0:
            per_rank_bytes = _get_per_rank_bytes(node, is_ag)
            if per_rank_bytes is not None and per_rank_bytes < min_bytes:
                skipped_small += 1
                log.debug(
                    "LC skip %s %s: size %d < min_bytes %d",
                    coll_type,
                    node.name,
                    per_rank_bytes,
                    min_bytes,
                )
                continue

        if not skip_overlap_check:
            # Skip collectives with no compute to hide behind
            if not _has_compute_bound_overlap(node, graph):
                skipped_no_overlap += 1
                log.debug(
                    "LC skip %s %s: no compute-bound overlap", coll_type, node.name
                )
                continue

            # Skip if other groups' NCCL collectives overlap on NVLink
            if _has_other_group_collectives(node, group_name, graph):
                skipped_nvlink_contention += 1
                log.debug(
                    "LC skip %s %s: overlaps other-group collectives (NVLink contention)",
                    coll_type,
                    node.name,
                )
                continue

        target = None
        if is_ag:
            target = _select_low_contention_all_gather_target(
                symm_mem,
                input_node=node.args[0],
                use_ag_ce_multicast=use_ag_ce_multicast,
            )

        if not _replace_collective(
            node,
            graph,
            symm_mem,
            is_ag,
            group_name,
            target=target,
        ):
            skipped_out_variant += 1
            continue
        replacements += 1

    log.info(
        "Replaced %d/%d FSDP collectives "
        "(skipped_small=%d, skipped_no_overlap=%d, "
        "skipped_nvlink_contention=%d, skipped_out_variant=%d, "
        "min_bytes=%d, skip_overlap_check=%s)",
        replacements,
        len(collectives),
        skipped_small,
        skipped_no_overlap,
        skipped_nvlink_contention,
        skipped_out_variant,
        min_bytes,
        skip_overlap_check,
    )


def _enable_symm_mem(group_name):
    # [FUNCTION DOC] HELPER: Ensure symmetric memory is enabled for a process group name. Returns True if already
    # enabled or enabling succeeds (FutureWarning suppressed); returns False, logging at debug level, if enabling
    # raises TypeError, RuntimeError or KeyError.
    """Try to enable symmetric memory for a group. Returns True on success."""
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )

    if is_symm_mem_enabled_for_group(group_name):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
        return True
    except (TypeError, RuntimeError, KeyError) as e:
        log.debug("LC: cannot enable symm_mem for group %s: %s", group_name, e)
        return False


def _has_multicast_support(device_index: int) -> bool:
    # [FUNCTION DOC] HELPER: Return whether the given CUDA device index supports multicast in symmetric memory.
    # Any exception (for example missing bindings) is swallowed and yields False.
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return bool(
            _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device_index)
        )
    except Exception:
        return False


def _select_low_contention_all_gather_target(
    symm_mem,
    input_node,
    use_ag_ce_multicast=False,
):
    # [FUNCTION DOC] HELPER: Pick the symm_mem all-gather op to use. Returns the plain low-contention all-gather
    # unless multicast copy-engine use is requested and the device (taken from the input node's fake value if it is a
    # CUDA tensor, else the current CUDA device) supports multicast; otherwise logs and falls back to the plain op.
    if not use_ag_ce_multicast:
        return symm_mem._low_contention_all_gather.default

    device_index = None
    input_val = input_node.meta.get("val")
    if isinstance(input_val, torch.Tensor) and input_val.device.type == "cuda":
        device_index = input_val.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    if _has_multicast_support(device_index):
        return symm_mem._low_contention_all_gather_ce_multicast.default
    log.info(
        "low_contention_all_gather_ce_multicast requested but multicast is not "
        "supported on device %d; falling through.",
        device_index,
    )

    return symm_mem._low_contention_all_gather.default


def _replace_collective(
    node,
    graph,
    symm_mem,
    is_ag,
    group_name,
    target=None,
) -> bool:
    # [FUNCTION DOC] HELPER: Replace one collective node with the chosen symm_mem op. Returns False (no change)
    # for all-gather with no target, or for an existing out-variant all-gather when the target is not the multicast
    # variant (the plain one has no out form). Reduce-scatter always uses the low-contention reduce-scatter taking
    # the reduce op. The new node is inserted before the old one, inherits its meta, takes over all uses, and the
    # old node is erased. Returns True on replacement.
    input_node = node.args[0]
    if is_ag:
        if target is None:
            return False
        if node.target is torch.ops._c10d_functional.all_gather_into_tensor_out.default:
            out = node.kwargs["out"]
            if target is symm_mem._low_contention_all_gather_ce_multicast.default:
                target = symm_mem._low_contention_all_gather_ce_multicast_out.default
            else:
                # The existing low-contention all-gather has no out variant.
                return False
            args = (input_node, group_name, out)
        else:
            args = (input_node, group_name)
    else:
        reduce_op = node.args[1]
        target = symm_mem._low_contention_reduce_scatter.default
        args = (input_node, reduce_op, group_name)

    with graph.inserting_before(node):
        new_node = graph.call_function(target, args=args)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)
    return True


def _get_per_rank_bytes(node, is_ag):
    # [FUNCTION DOC] HELPER: Compute per-rank message size in bytes from the input's fake tensor value. For
    # all-gather this is the input size; for reduce-scatter it is the input size divided by the group size (third
    # argument, must be a positive int). Returns None if the size cannot be determined.
    """Return per-rank message bytes for a collective, or None if unknown."""
    input_val = node.args[0].meta.get("val") if node.args else None
    if not isinstance(input_val, torch.Tensor):
        return None
    total_bytes = input_val.nelement() * input_val.element_size()
    if is_ag:
        return total_bytes
    # For RS, input is the full tensor; per-rank = total / group_size
    group_size = node.args[2] if len(node.args) > 2 else None
    if not isinstance(group_size, int) or group_size <= 0:
        return None
    return total_bytes // group_size


def _is_compute_or_contains_compute(node, graph_module):
    # [FUNCTION DOC] HELPER: Return True if the node is a compute node, or is a ControlDeps call whose
    # subgraph (a get_attr second argument resolving to a GraphModule on graph_module) contains a compute node.
    # Raises AssertionError if the get_attr target is not a string.
    """Check if a node is compute, including inside control_deps subgraphs."""
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    if is_compute_node(node):
        return True
    from torch._inductor.fx_passes.control_dependencies import ControlDeps

    if node.op == "call_function" and isinstance(node.target, ControlDeps):
        subgraph_node = node.args[1] if len(node.args) > 1 else None
        if isinstance(subgraph_node, torch.fx.Node) and subgraph_node.op == "get_attr":
            if not isinstance(subgraph_node.target, str):
                raise AssertionError(
                    f"expected str get_attr target, got {type(subgraph_node.target)}"
                )
            subgraph = getattr(graph_module, subgraph_node.target, None)
            if isinstance(subgraph, torch.fx.GraphModule):
                for n in subgraph.graph.nodes:
                    if is_compute_node(n):
                        return True
    return False


def _has_compute_bound_overlap(start_node, graph):
    # [FUNCTION DOC] HELPER: Return True if any compute node (including inside control-deps subgraphs) appears
    # in graph order strictly between the collective and its wait node. Returns False if no wait is found.
    """Check if compute-bound ops exist between collective start and wait."""
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    graph_module = graph.owning_module
    in_range = False
    for node in graph.nodes:
        if node is start_node:
            in_range = True
            continue
        if node is wait_node:
            break
        if in_range and _is_compute_or_contains_compute(node, graph_module):
            return True
    return False


def _has_other_group_collectives(start_node, group_name, graph):
    # [FUNCTION DOC] HELPER: Return True if a collective of a different process group lies in graph order
    # between the start node and its wait node (NVLink contention). Returns False if no wait is found.
    """Check if other groups' collectives overlap, competing for NVLink."""
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    in_range = False
    for node in graph.nodes:
        if node is start_node:
            in_range = True
            continue
        if node is wait_node:
            break
        if not in_range:
            continue
        info = _get_collective_info(node)
        if info is not None:
            _, other_group = info
            if other_group != group_name:
                log.debug(
                    "LC contention %s: found %s (group %s) between start/wait",
                    start_node.name,
                    node.name,
                    other_group,
                )
                return True
    return False


def _is_wait_tensor(node):
    # [FUNCTION DOC] HELPER: Return True if the node is a call_function that is the functional wait_tensor op,
    # or whose name contains wait_tensor (covers the public namespace and ControlDeps-wrapped waits).
    """Check if node is a wait_tensor op (direct or wrapped in ControlDeps)."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops._c10d_functional.wait_tensor.default:
        return True
    # Handles public namespace (c10d_functional.wait_tensor) and
    # ControlDeps-wrapped wait_tensor (from TBB manual scheduling)
    return "wait_tensor" in node.name


def _find_wait_for_collective(start_node):
    # [FUNCTION DOC] HELPER: Find the wait node for a collective start node, or None. Looks first at direct
    # users, then, for out-variant all-gather / reduce-scatter, at users of the out buffer keyword argument.
    # KNOWN ISSUE: the docstring lists a ControlDeps-wrapped pattern, handled only through the name match in
    # the wait check, and only when the wrapper is a direct user.
    """Find the wait_tensor node for a collective.

    Handles multiple graph patterns:
    1. Direct: start -> wait_tensor(start)
    2. _out variant: start(out=buf) -> wait_tensor(buf)
    3. ControlDeps-wrapped: start -> control_deps(wait_tensor_subgraph, start)
    """
    for user in start_node.users:
        if _is_wait_tensor(user):
            return user

    # For _out variants, check users of the out-buffer keyword argument.
    c10d = torch.ops._c10d_functional
    if start_node.target in (
        c10d.all_gather_into_tensor_out.default,
        c10d.reduce_scatter_tensor_out.default,
    ):
        out_buf = start_node.kwargs.get("out")
        if isinstance(out_buf, torch.fx.Node):
            for user in out_buf.users:
                if _is_wait_tensor(user):
                    return user

    return None
