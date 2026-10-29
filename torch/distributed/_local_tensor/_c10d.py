# [MODULE DOC]
# Local (single-process, simulated) implementations of c10d collectives and point-to-point ops
# operating on LocalTensor, dispatched to by LocalTensorMode. Provides both the "regular" flavor
# (takes a process group object, mutates preallocated outputs in place, returns a FakeWork handle)
# and the "functional" flavor (takes a group name or group, allocates and returns new outputs).
# Process group ranks are mapped to a layout so the same collective is replayed on every group
# that is a translated copy of the given group. Imports are side-effect free.
import functools
import math
import operator
from collections.abc import Callable, Sequence
from datetime import timedelta

import torch
from torch._C import ScriptObject
from torch._C._distributed_c10d import FakeWork, PythonCallbackWork
from torch.distributed._mesh_layout import _FlatLayout
from torch.distributed.distributed_c10d import (
    _check_op,
    _get_default_group,
    _resolve_process_group,
    GroupName,
    ProcessGroup,
    ReduceOp,
    Work,
)


# NOTE: Most of the c10d collectives often take a Tensor[] (or Tensor[][])
# when you would expect Tensor (or Tensor[]).  In fact, there will only ever
# be one Tensor in this case; the old signature was to support dispatching a
# collective on multiple devices (ala DataParallel) but we don't support that
# API anymore.  Note that we are not 100% consistent about this; some more
# modern collectives like _allgather_base_ got rid of the unnecessary list.
# When in doubt, consult the code that dispatches to the collective on the PG
# in distributed_c10d.py e.g., work = group.allgather([tensor_list], [tensor],
# opts) indicates it's always a list.


def _gcd_list(numbers: Sequence[int]) -> int:
    # [FUNCTION DOC] HELPER: Greatest common divisor of a sequence of ints; returns 0 for an empty sequence.
    return 0 if not numbers else functools.reduce(math.gcd, numbers)


def _indices_to_layout(indices: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # [FUNCTION DOC] HELPER: Infers a regular-grid layout (tuple of shapes, tuple of strides) from a sorted list of unique indices.
    # The last stride is the gcd of consecutive differences; the start of each row is any index whose predecessor at that stride is absent;
    # recurses on the row starts. Zero or one index gives an empty layout. Raises AssertionError if the stride cannot be determined or the indices do not form a regular grid.
    # Base case: A single index represents a point, not a dimension.
    if len(indices) <= 1:
        return (), ()

    # The smallest stride is likely the GCD of the differences between consecutive indices.
    # For a sorted, unique list, all differences will be positive.
    diffs = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
    last_stride = _gcd_list(diffs)

    # This case should not be reached if indices are unique and sorted.
    if last_stride == 0:
        raise AssertionError("Cannot determine stride; indices may not be unique.")

    # Identify the starting index of each "row" in the last dimension.
    # An index starts a new row if the preceding index (index - stride) is not present.
    indices_set = set(indices)
    higher_dim_indices = [indices[0]]
    for index in indices[1:]:
        if (index - last_stride) not in indices_set:
            higher_dim_indices.append(index)

    # From the number of rows, we can deduce the shape of the last dimension.
    if len(indices) % len(higher_dim_indices) != 0:
        raise AssertionError(
            "Indices do not form a regular grid. "
            f"Found {len(higher_dim_indices)} subgroups for {len(indices)} total elements."
        )
    last_shape = len(indices) // len(higher_dim_indices)

    # Recurse on the higher-dimensional indices (the start of each row).
    higher_shapes, higher_strides = _indices_to_layout(higher_dim_indices)

    # Combine the results from the recursion with the current dimension's results.
    final_shapes = higher_shapes + (last_shape,)
    final_strides = higher_strides + (last_stride,)

    return final_shapes, final_strides


def _resolve_pg_or_name(
    group: ScriptObject | ProcessGroup | str | GroupName,
) -> ScriptObject | ProcessGroup:
    # [FUNCTION DOC] HELPER: Accepts a live ProcessGroup, a ScriptObject group, or a group name; returns the first two unchanged and
    # resolves a name to its process group (needed when functional collectives receive a real group instead of a name).
    """Accept any of the group reps that may reach a functional collective.

    Under ``compile_on_one_rank=True`` the functional collective ops receive
    the live ``ProcessGroup`` instead of a string group name, so we cannot
    unconditionally call ``_resolve_process_group``.
    """
    if isinstance(group, (ProcessGroup, ScriptObject)):
        return group
    # ``GroupName`` is ``NewType("GroupName", str)``, so wrapping ``group`` here
    # is a runtime no-op. The cast exists purely so pyrefly accepts the
    # narrowed ``str | GroupName`` argument against the declared parameter type.
    return _resolve_process_group(GroupName(group))


def _prepare_collective_groups(
    process_group_so: ScriptObject | ProcessGroup,
) -> tuple[list[int], list[int], int]:
    # [FUNCTION DOC] HELPER: Given a process group (unboxing a ScriptObject), returns a triple of (group ranks shifted so the first is zero, the starting
    # global offsets of every translated copy of this group across the default world, and the original first rank). The group ranks must be non-empty and sorted
    # (AssertionError otherwise); the layout is inferred from the ranks and its complement over the world size gives the offsets.
    process_group = (
        ProcessGroup.unbox(process_group_so)
        if isinstance(process_group_so, ScriptObject)
        else process_group_so
    )

    ranks = torch.distributed.get_process_group_ranks(process_group)
    if not ranks:
        raise AssertionError
    # TODO: We can handle permutations but the layout inference algorithm will
    # lose the permutation so we will have to reapply it
    if ranks != sorted(ranks):
        raise AssertionError(ranks)
    offset = ranks[0]
    ranks = [r - offset for r in ranks]

    shape, strides = _indices_to_layout(ranks)
    layout = _FlatLayout(shape, strides)

    global_pg = _get_default_group()
    group_offsets = layout.complement(global_pg.size()).all_ranks_from_zero()

    return ranks, group_offsets, offset


# NB: There are two flavors of the collectives: regular and functional. Regular collectives
# allocate outputs to write the result to, accept process group and support async ops (return
# work object). Functional collectives expect the implementation to allocate outputs, accept
# process group name that must be resolved and do not support async ops (return output).
def _local_functional_all_gather_into_tensor(
    tensor: torch.Tensor,
    group_size: int,
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Functional all_gather_into_tensor on a LocalTensor. For each group copy whose ranks are all present, concatenates the members' shards along dimension 0
    # and gives every member a clone. Returns a new LocalTensor. Raises AssertionError if the input is not a LocalTensor. The group_size argument is unused.
    # "all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor"
    from . import LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        gathered_tensor = torch.cat(group_tensors, dim=0)

        for rank in group_ranks:
            output_local_tensors[rank] = gathered_tensor.clone()

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_reduce_scatter_tensor(
    tensor: torch.Tensor,
    reduce_op: str,
    group_size: int,
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Functional reduce_scatter_tensor on a LocalTensor. Per group copy, reduces the members' shards with the named reduce op, splits the result along dimension 0
    # into equal chunks and gives chunk i to member i (a zero-sized tensor if there are fewer chunks than members). Returns a new LocalTensor; AssertionError if input is not a LocalTensor.
    #  "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor"
    from . import _zero_sized_like, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        reduced_tensor = _local_reduce(reduce_op, group_tensors)

        scattered_tensor = torch.split(
            reduced_tensor,
            reduced_tensor.size(0) // len(group_ranks),
            dim=0,
        )

        for i, rank in enumerate(group_ranks):
            if i < len(scattered_tensor):
                output_local_tensors[rank] = scattered_tensor[i].clone()
            else:
                output_local_tensors[rank] = _zero_sized_like(reduced_tensor, 0)

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_shard_dim_alltoall(
    tensor: torch.Tensor,
    gather_dim: int,
    shard_dim: int,
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Functional shard-dim all-to-all on a LocalTensor. Per group copy, concatenates the members' shards along gather_dim, splits that along shard_dim
    # into one chunk per member and gives chunk i to member i (zero-sized if missing). Returns a new LocalTensor; AssertionError if input is not a LocalTensor.
    # "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor"
    from . import _zero_sized_like, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        gathered_tensor = torch.cat(group_tensors, dim=gather_dim)

        split_tensor = torch.split(
            gathered_tensor,
            gathered_tensor.size(shard_dim) // len(group_ranks),
            dim=shard_dim,
        )

        for i, rank in enumerate(group_ranks):
            if i < len(split_tensor):
                output_local_tensors[rank] = split_tensor[i].clone()
            else:
                output_local_tensors[rank] = _zero_sized_like(
                    gathered_tensor, shard_dim
                )

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_all_to_all_single(
    tensor: torch.Tensor,
    output_split_sizes: list[torch.SymInt],
    input_split_sizes: list[torch.SymInt],
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Functional all_to_all_single on a LocalTensor. Input split sizes (ints or per-rank SymInts) give each rank's chunk sizes along dimension 0;
    # within each group copy, member dst receives the dst-th chunk from every member (in group order), concatenated. Output split sizes are ignored. Returns a new LocalTensor;
    # AssertionError if the input is not a LocalTensor.
    # "all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor"
    from . import LocalIntNode, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    split_local_sizes: dict[int, list[int]] = {}
    for input_split_size in input_split_sizes:
        if isinstance(input_split_size, torch.SymInt) and isinstance(
            input_split_size.node, LocalIntNode
        ):
            local_ints = dict(input_split_size.node._local_ints.items())
        else:
            local_ints = {rank: int(input_split_size) for rank in tensor._local_tensors}
        for rank, split_size in local_ints.items():
            if rank not in split_local_sizes:
                split_local_sizes[rank] = []
            split_local_sizes[rank].append(split_size)

    split_local_tensors: dict[int, list[torch.Tensor]] = {}

    for rank, split_sizes in split_local_sizes.items():
        split_local_tensors[rank] = list(
            torch.split(tensor._local_tensors[rank], split_sizes)
        )

    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        if not all(rank in split_local_tensors for rank in group_ranks):
            continue

        for i, dst in enumerate(group_ranks):
            splits = []
            for src in group_ranks:
                splits.append(split_local_tensors[src][i])
            output_local_tensors[dst] = torch.cat(splits)

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_broadcast_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    root_rank: int,
    root_tensor: int,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # [FUNCTION DOC] HELPER: Local broadcast_. Takes a one-element tensor list (root_tensor must be zero, else AssertionError) holding a LocalTensor. In each fully-present group copy,
    # copies the root rank's shard into every other member's shard in place. Returns the tensors and a boxed FakeWork.
    # "broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int root_rank, int root_tensor, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)"
    from . import LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    if root_tensor != 0:
        raise AssertionError
    tensor = tensors[0]

    ranks, group_offsets, offset = _prepare_collective_groups(process_group_so)

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the broadcast on them
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        source_rank = group_offset + relative_root_rank
        source_tensor = tensor._local_tensors[source_rank]

        # Broadcast the source tensor to all ranks in this group
        for rank in group_ranks:
            if source_rank != rank:
                tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (tensors, work_so)


def _local_reduce(
    reduce_op: ReduceOp | str,
    tensors: list[torch.Tensor],
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Reduces a list of tensors element-wise using the given ReduceOp or its lowercase string name (sum, avg, product, min, max, band, bor, bxor).
    # Average is a sum divided by the count. Raises NotImplementedError for premul_sum or an unknown op.
    if reduce_op == ReduceOp.SUM or reduce_op == "sum":
        op = operator.add
    elif reduce_op == ReduceOp.AVG or reduce_op == "avg":
        op = None
    elif reduce_op == ReduceOp.PRODUCT or reduce_op == "product":
        op = operator.mul
    elif reduce_op == ReduceOp.MIN or reduce_op == "min":
        op = torch.minimum
    elif reduce_op == ReduceOp.MAX or reduce_op == "max":
        op = torch.maximum
    elif reduce_op == ReduceOp.BAND or reduce_op == "band":
        op = torch.bitwise_and
    elif reduce_op == ReduceOp.BOR or reduce_op == "bor":
        op = torch.bitwise_or
    elif reduce_op == ReduceOp.BXOR or reduce_op == "bxor":
        op = torch.bitwise_xor
    elif reduce_op == ReduceOp.PREMUL_SUM or reduce_op == "premul_sum":
        raise NotImplementedError("PREMUL_SUM: need to add binding for scaling factor")
    else:
        raise NotImplementedError(f"ReduceOp {reduce_op} not implemented")

    if reduce_op == ReduceOp.AVG or reduce_op == "avg":
        return functools.reduce(operator.add, tensors) / len(tensors)
    else:
        if op is None:
            raise AssertionError
        return functools.reduce(op, tensors)


def _local_all_reduce_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    sparse_indices: torch.Tensor | None = None,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # [FUNCTION DOC] HELPER: Local allreduce_. Takes a one-element tensor list holding a LocalTensor (AssertionError otherwise); for each fully-present group copy reduces the members' shards
    # and writes the result back to every member in place. Returns the tensors and a boxed FakeWork.
    # "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? sparse_indices, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
    from . import LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]
    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on them
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        # Collect tensors from the specified ranks in this group
        group_tensors = []
        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        # Perform the reduction operation
        reduced_tensor = _local_reduce(reduce_op, group_tensors)

        # Update all tensors in the group with the reduced result
        for rank in group_ranks:
            tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (tensors, work_so)


def _local_allreduce_coalesced_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local allreduce_coalesced_. Like local allreduce but over several LocalTensors (AssertionError for non-LocalTensors); each is reduced independently per group copy, in place.
    # Returns a boxed FakeWork.
    # "allreduce_coalesced_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for tensor in tensors:
            if not isinstance(tensor, LocalTensor):
                raise AssertionError("Input tensor must be a LocalTensor")
            if not all(rank in tensor._local_tensors for rank in group_ranks):
                continue
            # Collect tensors from the specified ranks in this group
            group_tensors = []
            for rank in group_ranks:
                group_tensors.append(tensor._local_tensors[rank])

            # Perform the reduction operation
            reduced_tensor = _local_reduce(reduce_op, group_tensors)

            # Update all tensors in the group with the reduced result
            for rank in group_ranks:
                tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_reduce_scatter_tensor_coalesced_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local reduce_scatter_tensor_coalesced_. For each (input, output) pair of LocalTensors and each fully-present group copy, reduces the members' inputs, splits
    # along dimension 0 into one chunk per member and copies chunk i into member i's output in place. Returns a boxed FakeWork. AssertionError for non-LocalTensors.
    # "reduce_scatter_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, "
    # "int timeout=-1) -> __torch__.torch.classes.c10d.Work"

    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            if not isinstance(input_tensor, LocalTensor):
                raise AssertionError("Input tensor must be a LocalTensor")
            if not isinstance(output_tensor, LocalTensor):
                raise AssertionError("Output tensor must be a LocalTensor")
            if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                continue
            if not all(rank in output_tensor._local_tensors for rank in group_ranks):
                continue

            # Collect tensors from the specified ranks in this group
            group_inputs = []
            for rank in group_ranks:
                group_inputs.append(input_tensor._local_tensors[rank])

            # Perform the reduction operation
            reduced_input = _local_reduce(reduce_op, group_inputs)

            reduced_input_splits = torch.split(
                reduced_input, reduced_input.size(0) // len(group_ranks), dim=0
            )

            # Update all tensors in the group with the reduced result
            for i, rank in enumerate(group_ranks):
                output_tensor._local_tensors[rank].copy_(reduced_input_splits[i])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_allgather_base_(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[torch.Tensor, ScriptObject]:
    # [FUNCTION DOC] HELPER: Local _allgather_base_. For each group copy present in both tensors, concatenates the members' input shards along dimension 0 and copies the result into every
    # member's output shard in place. Returns the output tensor and a boxed FakeWork. AssertionError for non-LocalTensors.
    # "_allgather_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup
    # process_group, bool async_op=True, int timeout=-1) -> (Tensor, __torch__.torch.classes.c10d.Work)");
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    if not isinstance(input_tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        gathered_tensors = []
        for rank_i in group_ranks:
            gathered_tensors.append(input_tensor._local_tensors[rank_i])

        gathered_tensor = torch.cat(gathered_tensors, dim=0)

        for rank_i in group_ranks:
            output_tensor._local_tensors[rank_i].copy_(gathered_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return output_tensor, work_so


def _local_reduce_scatter_base_(  # type: ignore[no-untyped-def]
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[torch.Tensor, ScriptObject]:
    # [FUNCTION DOC] HELPER: Local _reduce_scatter_base_. For each group copy present in both tensors, reduces the members' input shards, splits along dimension 0 into one chunk per member
    # and copies chunk i into member i's output shard in place. Returns the output tensor and a boxed FakeWork. AssertionError for non-LocalTensors.
    # "_reduce_scatter_base_(Tensor output_tensor, Tensor input_tensor,
    # __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op,
    # bool async_op=True, int timeout=-1) -> (Tensor, __torch__.torch.classes.c10d.Work)"

    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    if not isinstance(input_tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        gathered_tensors = []
        for rank_i in group_ranks:
            gathered_tensors.append(input_tensor._local_tensors[rank_i])

        reduced_tensor = _local_reduce(reduce_op, gathered_tensors)

        scattered_tensor = torch.split(
            reduced_tensor,
            reduced_tensor.size(0) // len(group_ranks),
            dim=0,
        )

        for i, rank_i in enumerate(group_ranks):
            output_tensor._local_tensors[rank_i].copy_(scattered_tensor[i].clone())

    work = FakeWork()
    work_so = Work.boxed(work)
    return output_tensor, work_so


def _local_all_gather_(
    output_tensors: list[list[torch.Tensor]],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[list[torch.Tensor]], ScriptObject]:
    # [FUNCTION DOC] HELPER: Local allgather_ (list-of-list flavor; exactly one output list and one input tensor, AssertionError otherwise). For each group copy, the i-th output LocalTensor receives
    # the i-th member's input (the input may be a plain tensor, used as is). Returns the outputs and a boxed FakeWork. AssertionError if an output is not a LocalTensor.
    # "allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[][], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    if len(output_tensors) != 1:
        raise AssertionError
    if len(input_tensors) != 1:
        raise AssertionError

    input_tensor = input_tensors[0]
    # pyrefly: ignore [bad-assignment]
    output_tensors = output_tensors[0]

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for i in range(len(output_tensors)):
        if not isinstance(output_tensors[i], LocalTensor):
            raise AssertionError("Output tensor must be a LocalTensor")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the all_gather on them
        group_ranks = [group_offset + r for r in ranks]

        # For each rank in the group, gather from their input tensor
        for i, rank_i in enumerate(group_ranks):
            # allgather object happens to create pure tensor, so we special case it here
            source_tensor = input_tensor
            if isinstance(input_tensor, LocalTensor):
                source_tensor = input_tensor._local_tensors[rank_i]
            # pyrefly: ignore [missing-attribute]
            output_tensors[i].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    # pyrefly: ignore [bad-return]
    return ([output_tensors], work_so)


def _local_allgather_into_tensor_coalesced_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local allgather_into_tensor_coalesced_. For each pair of input and output LocalTensors and each fully-present group copy, concatenates the members' inputs along dimension 0
    # and copies it into every member's output in place. Raises AssertionError on length mismatch or non-LocalTensors. Returns a boxed FakeWork.
    # "allgather_into_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True) "
    # "-> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    # Each output tensor should be sized to hold all gathered inputs
    # outputs[i] will contain all inputs[i] from all ranks
    if len(output_tensors) != len(input_tensors):
        raise AssertionError(
            f"Number of outputs ({len(output_tensors)}) must match "
            f"number of inputs ({len(input_tensors)})"
        )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allgather_into_tensor on them
        group_ranks = [group_offset + r for r in ranks]

        # For each input/output pair
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            if not isinstance(input_tensor, LocalTensor):
                raise AssertionError("Input tensor must be a LocalTensor")
            if not isinstance(output_tensor, LocalTensor):
                raise AssertionError("Output tensor must be a LocalTensor")

            if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                continue
            if not all(rank in output_tensor._local_tensors for rank in group_ranks):
                continue

            # Gather input_tensor from all ranks into output_tensor
            # The output should be a concatenation of all inputs along the first dimension
            gathered_tensors = []
            for rank in group_ranks:
                gathered_tensors.append(input_tensor._local_tensors[rank])

            # Concatenate along first dimension and copy to output
            if gathered_tensors:
                concatenated = torch.cat(gathered_tensors, dim=0)
                for rank in group_ranks:
                    output_tensor._local_tensors[rank].copy_(concatenated)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_gather_(
    output_tensors: list[list[torch.Tensor]],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    root_rank: int,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Unsupported: always raises NotImplementedError because gather only delivers data to the root rank (an MPMD pattern); use allgather instead.
    # "gather_(Tensor[][] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, "
    # "bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"
    raise NotImplementedError(
        "LocalTensor does not support MPMD operations like gather "
        "(only root rank receives data). Use SPMD collective operations like allgather instead."
    )


def _local_scatter_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[list[torch.Tensor]],
    process_group_so: ScriptObject,
    root_rank: int,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # [FUNCTION DOC] HELPER: Local scatter_. Requires one output LocalTensor and one input list whose length equals the group size (AssertionError otherwise). In each fully-present group copy,
    # member i receives the i-th input's shard taken from the root rank (assumed to sit at the same relative position in every copy). Returns the outputs and a boxed FakeWork.
    # "scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, "
    # "bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    if len(output_tensors) != 1:
        raise AssertionError
    if len(input_tensors) != 1:
        raise AssertionError
    output_tensor = output_tensors[0]
    # pyrefly: ignore [bad-assignment]
    input_tensors = input_tensors[0]

    ranks, group_offsets, offset = _prepare_collective_groups(process_group_so)

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    if len(ranks) != len(input_tensors):
        raise AssertionError((ranks, input_tensors))

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the scatter on them
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        # Root rank scatters its input tensors to all ranks in this group
        for i, rank in enumerate(group_ranks):
            input_tensor = input_tensors[i]
            if not isinstance(input_tensor, LocalTensor):
                raise AssertionError
            # Each rank i gets the i-th input tensor from the root
            source_tensor = input_tensor._local_tensors[
                group_offset + relative_root_rank
            ]
            output_tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # [FUNCTION DOC] HELPER: Local alltoall_. Lists of input and output LocalTensors must each have one entry per group member (AssertionError otherwise). In each group copy, the j-th input of member i
    # is copied to member j's i-th output, in place. Returns the outputs and a boxed FakeWork.
    # "alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)";

    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not (len(input_tensors) == len(output_tensors) == len(ranks)):
        raise AssertionError(
            f"Number of input tensors ({len(input_tensors)}), "
            f"output tensors ({len(output_tensors)}), and ranks ({len(ranks)}) must match"
        )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the alltoall on them
        group_ranks = [group_offset + r for r in ranks]

        # In alltoall, rank i sends input_tensors[j] to rank j and receives into output_tensors[i] from rank j
        for i, rank_i in enumerate(group_ranks):
            output_tensor = output_tensors[i]
            if not isinstance(output_tensor, LocalTensor):
                raise AssertionError("Output tensor must be a LocalTensor")

            if not all(rank in output_tensor._local_tensors for rank in group_ranks):
                continue

            for j, rank_j in enumerate(group_ranks):
                input_tensor = input_tensors[j]
                if not isinstance(input_tensor, LocalTensor):
                    raise AssertionError("Input tensor must be a LocalTensor")

                if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                    continue

                # Rank i's j-th input tensor goes to rank j's i-th output tensor
                source_tensor = input_tensor._local_tensors[rank_i]
                output_tensor._local_tensors[rank_j].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_base_(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local alltoall_base_. Input and output must be LocalTensors (AssertionError otherwise). In each fully-present group copy, splits each member's input along dimension 0
    # (by input split sizes, or evenly if none) and copies split j of member i into member j's output at the region chosen by output split sizes (or even regions), in place.
    # Raises ValueError if a split and its destination region differ in element count. Returns a boxed FakeWork.
    # "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] output_split_sizes, int[] input_split_sizes, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(input_tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    # Convert split sizes to lists if they aren't already
    if output_split_sizes is not None:
        output_split_sizes = list(output_split_sizes)
    if input_split_sizes is not None:
        input_split_sizes = list(input_split_sizes)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the alltoall_base on them
        group_ranks = [group_offset + r for r in ranks]

        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        for i, rank_i in enumerate(group_ranks):
            # Split input tensor from rank_i according to input_split_sizes
            rank_tensor = input_tensor._local_tensors[rank_i]

            if input_split_sizes is not None and len(input_split_sizes) > 0:
                # Split the input tensor
                input_splits = torch.split(rank_tensor, input_split_sizes, dim=0)
            else:
                # No split sizes specified, split evenly
                split_size = rank_tensor.size(0) // len(group_ranks)
                input_splits = torch.split(rank_tensor, split_size, dim=0)

            # Send each split to the corresponding rank
            for j, rank_j in enumerate(group_ranks):
                if j < len(input_splits):
                    split_tensor = input_splits[j]

                    # Determine where to place this split in the output tensor
                    if output_split_sizes is not None and len(output_split_sizes) > 0:
                        # Calculate offset based on output split sizes
                        output_offset = sum(output_split_sizes[:i]) if i > 0 else 0
                        end_offset = (
                            output_offset + output_split_sizes[i]
                            if i < len(output_split_sizes)
                            else output_tensor._local_tensors[rank_j].size(0)
                        )
                    else:
                        # No output split sizes, use even splits
                        split_size = output_tensor._local_tensors[rank_j].size(
                            0
                        ) // len(group_ranks)
                        output_offset = i * split_size
                        end_offset = min(
                            (i + 1) * split_size,
                            output_tensor._local_tensors[rank_j].size(0),
                        )

                    # Copy the split to the appropriate section of the output tensor
                    output_section = output_tensor._local_tensors[rank_j][
                        output_offset:end_offset
                    ]
                    if split_tensor.numel() != output_section.numel():
                        raise ValueError(
                            f"all_to_all_single: input split from rank {rank_i} to "
                            f"rank {rank_j} has {split_tensor.numel()} elements, but "
                            f"the corresponding output split has "
                            f"{output_section.numel()} elements"
                        )
                    if output_section.numel() > 0:
                        # Reshape split_tensor to match output_section if necessary
                        if split_tensor.size() != output_section.size():
                            split_tensor = split_tensor.view(output_section.size())
                        output_section.copy_(split_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_barrier(
    tensor: torch.Tensor,
    process_group_so: ScriptObject,
    device_ids: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local barrier: only checks the tensor is a LocalTensor (AssertionError otherwise); no synchronization is needed in one process. Returns a boxed FakeWork.
    # "barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] device_ids, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    # Barrier is a synchronization primitive - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    if not isinstance(tensor, LocalTensor):
        raise AssertionError

    # In a real distributed setting, barrier would synchronize all processes
    # In local simulation, this is essentially a no-op since all ranks are local
    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_monitored_barrier_(
    tensor: torch.Tensor,
    process_group_so: ScriptObject,
    device_ids: list[int],
    timeout: int,
    wait_all_ranks: bool,
) -> None:
    # [FUNCTION DOC] HELPER: Local monitored_barrier_: only checks the tensor is a LocalTensor (AssertionError otherwise); otherwise a no-op returning nothing.
    # "monitored_barrier_(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] device_ids, int timeout, bool wait_all_ranks) -> ()";

    from . import LocalTensor

    # Monitored barrier is a synchronization primitive with monitoring - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    if not isinstance(tensor, LocalTensor):
        raise AssertionError

    # In a real distributed setting, monitored barrier would synchronize all processes
    # and provide monitoring capabilities. In local simulation, this is essentially a no-op
    # since all ranks are local and no actual synchronization is needed
    return


def _local_send(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    dst: int,
    tag: int,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local send, valid only under LocalRunnerMode. Takes one LocalTensor tagged with its source rank (see _attach_rank), and queues that rank's shard for the destination
    # through the current runner. Raises AssertionError for a wrong tensor count or type. Returns a boxed FakeWork.
    # "send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int dst, int tag) -> __torch__.torch.classes.c10d.Work";

    from . import LocalRunnerMode, LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a Tensor")
    src = int(tensor.__src_rank__)

    LocalRunnerMode.current()._signal_send(src, dst, tensor._local_tensors[src])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_recv_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    src: int,
    tag: int,
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local recv_, valid only under LocalRunnerMode. Takes one LocalTensor tagged with its rank (see _attach_rank) and returns a boxed work whose completion callback waits on the
    # current runner until data arrives from src (or from any source if src is -1) and then stores it as that rank's shard. Raises AssertionError for a wrong tensor count or type.
    # "recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int src, int tag) -> __torch__.torch.classes.c10d.Work";
    from . import LocalRunnerMode, LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a Tensor")
    dst = int(tensor.__src_rank__)

    def _recv_and_store(timeout: timedelta) -> bool:
        # [FUNCTION DOC] HELPER: Work callback: blocks via the current runner until data is available, stores it, and returns True. The timeout argument is unused.
        def _wait_and_store(obj: object) -> None:
            # [FUNCTION DOC] HELPER: Stores the received object as this rank's shard; raises AssertionError if it is not a tensor.
            if not isinstance(obj, torch.Tensor):
                raise AssertionError("Expected to receive a Tensor")
            if not isinstance(tensor, LocalTensor):
                raise AssertionError("Input tensor must be a Tensor")
            tensor._local_tensors[dst] = obj

        LocalRunnerMode.current()._wait_recv(src, dst, _wait_and_store)
        return True

    work = PythonCallbackWork(_recv_and_store)
    work_so = Work.boxed(work)
    return work_so


def _local_recv_any_source_(
    tensors: list[torch.Tensor], process_group_so: ScriptObject, tag: int
) -> ScriptObject:
    # [FUNCTION DOC] HELPER: Local recv_any_source_: same as local recv with source rank -1 (any sender).
    # "recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int tag) -> __torch__.torch.classes.c10d.Work";

    return _local_recv_(tensors, process_group_so, -1, tag)


def _attach_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Tags a tensor (unwrapping DTensor to its local tensor) with the rank that initiates a send or receive, so the local p2p implementations can find it. Mutates the tensor by
    # setting an attribute; returns the (possibly unwrapped) tensor.
    """
    Attaches rank as an attribute to given tensor so that the send or recv implementation
    knows which rank initiates the operation (note under local tensor mode ).
    """
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor

    tensor.__src_rank__ = rank  # type: ignore[attr-defined]
    return tensor


def local_p2p_op(
    dst: torch.SymInt,
    tensor: torch.Tensor,
    op: Callable[[torch.Tensor, int], Work | None],
) -> Work | list[Work | None] | None:
    # [FUNCTION DOC] Runs a point-to-point op for every simulated (source, destination) pair, where dst is a SymInt backed by a LocalIntNode mapping source rank to destination rank
    # (AssertionError otherwise). For each pair tags the tensor with the source rank and calls op with the tensor and destination. Returns the list of resulting Work objects (or None entries).
    # Validates op with the c10d helper first.
    """
    Runs a point-to-point (P2P) operation for all combinations of source and destination ranks.
    """
    _check_op(op)

    from . import LocalIntNode

    if not isinstance(dst.node, LocalIntNode):
        raise AssertionError(
            "Expected 'dst' to be a LocalIntNode where the value is the "
            "destination rank and key is the source rank"
        )

    w = []
    for s, d in dst.node._local_ints.items():
        tensor = _attach_rank(tensor, s)
        w.append(op(tensor, d))
    return w


def wait_all(work: Work | list[Work | None] | None) -> None:
    # [FUNCTION DOC] Waits for a Work, or for each non-None Work in a list; does nothing for None.
    """
    Waits for all work objects in the input to complete.

    A single Work object, None, or a list of Work objects (possibly containing None).
    If None, does nothing. If a single Work, waits for it to complete. If a list, waits
    for each non-None Work in the list to complete.
    """

    if work is None:
        return
    if isinstance(work, Work):
        work = [work]
    for w in work:
        if w is None:
            continue
        w.wait()
