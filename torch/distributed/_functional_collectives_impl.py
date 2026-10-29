# mypy: allow-untyped-defs
# [MODULE DOC] Implementations of the legacy (c10d_functional) functional collectives: each
# helper resolves the group name from ranks and tag, then forwards to the native
# _c10d_functional operator. No import-time side effects beyond imports.

import torch
import torch.distributed.distributed_c10d as c10d


"""
This file contains the op impls for the legacy (c10d_functional) functional collectives.
These impls simply call into the native (_c10d_functional) functional collectives.
"""


def _broadcast(input, src, tag, ranks, group_size):
    # [FUNCTION DOC] HELPER: Resolves the group by ranks and tag and calls native broadcast
    # from src; returns the resulting tensor (group_size unused).
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.broadcast(
        input,
        src,
        group_name,
    )


def _all_reduce(input, reduce_op, tag, ranks, group_size):
    # [FUNCTION DOC] HELPER: Resolves the group and calls native all_reduce with reduce_op;
    # returns the reduced tensor (group_size unused).
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce(
        input,
        reduce_op,
        group_name,
    )


def _all_reduce_coalesced(inputs, reduce_op, tag, ranks, group_size):
    # [FUNCTION DOC] HELPER: Resolves the group and calls native all_reduce_coalesced on the
    # list of tensors; returns the list of reduced tensors (group_size unused).
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce_coalesced(
        inputs,
        reduce_op,
        group_name,
    )


def _all_gather_into_tensor(input, tag, ranks, group_size):
    # [FUNCTION DOC] HELPER: Resolves the group and calls native all_gather_into_tensor with
    # group_size; returns the gathered tensor.
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor(
        input,
        group_size,
        group_name,
    )


def _all_gather_into_tensor_coalesced(input, tag, ranks, group_size):
    # [FUNCTION DOC] HELPER: Resolves the group and calls the native coalesced all-gather on a
    # list of tensors with group_size; returns the list of gathered tensors.
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        input,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor(
    input: torch.Tensor,
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    # [FUNCTION DOC] HELPER: Resolves the group and calls native reduce_scatter_tensor with
    # reduce_op and group_size; returns this rank's reduced shard.
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduce_op,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor_coalesced(
    inputs: list[torch.Tensor],
    reduce_op: str,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    # [FUNCTION DOC] HELPER: Resolves the group and calls the native coalesced
    # reduce-scatter on a list of tensors; returns the list of reduced shards.
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs,
        reduce_op,
        group_size,
        group_name,
    )


def _all_to_all_single(
    input: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    tag: str,
    ranks: list[int],
    group_size: int,
):
    # [FUNCTION DOC] HELPER: All-to-all on the first dimension. Split size lists must be both
    # given or both None (else AssertionError); if None, an even split of the first dimension
    # by group_size is used for both. Resolves the group and calls native all_to_all_single;
    # returns the output tensor.
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [input.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_to_all_single(
        input,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )


def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Calls native wait_tensor to block until the async collective that
    # produced the tensor completes; returns the tensor.
    return torch.ops._c10d_functional.wait_tensor(tensor)


def _isend(tensor: torch.Tensor, dst: int, tag: str, group_name):
    # [FUNCTION DOC] HELPER: Calls native isend to send tensor to rank dst in group_name with tag.
    return torch.ops._c10d_functional.isend(tensor, dst, tag, group_name)


def _irecv(tensor: torch.Tensor, src: int, tag: str, group_name):
    # [FUNCTION DOC] HELPER: Calls native irecv to receive into tensor from rank src in group_name.
    return torch.ops._c10d_functional.irecv(tensor, src, tag, group_name)


def _batch_p2p_ops(
    op_list: list[str],
    peer_list: list[int],
    tag_list: list[int],
    tensors: list[torch.Tensor],
    group_name: str,
):
    # [FUNCTION DOC] HELPER: Calls native batch_p2p_ops with parallel lists of operation names,
    # peer ranks, tags and tensors for one group; returns its result.
    return torch.ops._c10d_functional.batch_p2p_ops(
        op_list, peer_list, tag_list, tensors, group_name
    )
