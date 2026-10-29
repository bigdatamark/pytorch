# mypy: allow-untyped-defs
# [MODULE DOC] Sharded implementation of torch.nn.functional.embedding_bag for
# a weight held as a ShardedTensor with a ChunkShardingSpec. Provides row-wise
# (dim 0) and column-wise (dim 1) paths, argument validation, and a helper that
# gathers ids, per-sample weights and offsets in one collective.
# Import-time side effect: the decorator registers sharded_embedding_bag as the
# custom sharding spec op for ChunkShardingSpec and functional embedding_bag.

from typing import cast

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter

from ._common import (
    _all_gather_base_input,
    _handle_col_wise_sharding_base,
    _handle_max_norm_col_wise,
    _handle_row_wise_mask,
)


@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg):
    # [FUNCTION DOC] Torch-function handler for embedding_bag with a sharded
    # weight. args holds the id tensor and ShardedTensor weight; kwargs may carry
    # offsets, per_sample_weights, mode, max_norm, norm_type, include_last_offset
    # and padding_idx. Validates, drops the trailing offset when
    # include_last_offset is set, then dispatches on sharding dim: 1 uses the
    # column-wise path (writing the possibly renormed shard back into the
    # weight's first local shard), 0 uses the row-wise path. Returns the
    # aggregated bag output. Raises RuntimeError for any other sharding dim.
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.
    This method computes a sharded embedding bag aggregation and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 4 shard of (4 x 17).
    The algorithm is as follows:

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
    3. If ``max_norm`` is specified, the extra row guarantees that the mask ID will
       not affect the behavior of weight re-norm.
    4. The example above only happens in one rank and each rank does a very similar thing.
       For "Mean" mode we need to divide by either column size (2D) or the interval length
       defined by the offset (excluding the row specified in ``padding_idx``).
       We also need to mask the nonexistent row to neg Inf so that negative value does not
       gets wiped out in the "Max" mode.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding bag operation under the given mode by
       apply each input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 4) ((2 x 4) for the last) matrices on each rank.
       We transpose the aggregation result.
    3. Next, we concatenate these 4 matrices and perform an all2all to share the
       appropriate (5 x 4) or (2 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 4) matrix and finally we transpose the output again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    # Validate input params
    _validate_embedding_bag_param(args, kwargs)

    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    include_last_offset = kwargs.get("include_last_offset")
    padding_idx = kwargs.get("padding_idx")

    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if include_last_offset:
        offsets = offsets[:-1]

    if sharding_dim == 1:
        output, local_shard = _handle_col_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            pg,
        )
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    else:
        raise RuntimeError(
            f"nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!"
        )


def _validate_embedding_bag_param(args, kwargs):
    # [FUNCTION DOC] HELPER: Checks embedding_bag arguments, returns nothing.
    # Raises TypeError for wrong tensor types (input, offsets, per_sample_weights,
    # weight); ValueError for input over 2 dims, non-2D weight, out-of-range ids,
    # inconsistent offsets or per_sample_weights shapes, unsupported mode (default
    # is mean; allowed sum, mean, max), bad include_last_offset usage,
    # non-positive max_norm, non-chunk spec, or not exactly one local shard;
    # RuntimeError if scale_grad_by_freq or sparse is set.
    # KNOWN ISSUE: the out-of-range ValueErrors pass printf-style args as extra
    # exception arguments without formatting, and report dim 1 size as the bound.
    # The error text for scale_grad_by_freq and sparse says nn.Embedding.
    """
    Validate input params of sharded embeddingBag op.

    Args:
        input: list of ID used for lookup and aggregation.
        weight: sharded weight tensor.
        kwargs: same as normal EmbeddingBag.

    Return: None.
    """

    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")
    include_last_offset = kwargs.get("include_last_offset")

    # Validate types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if offsets is not None and not isinstance(offsets, torch.Tensor):
        raise TypeError("offsets need to be torch.Tensor")
    if per_sample_weights is not None and not isinstance(
        per_sample_weights, torch.Tensor
    ):
        raise TypeError("per_sample_weights need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    if len(input.size()) > 2:
        raise ValueError("Input more than 2 dims not supported")
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")
    if int(torch.min(input).item()) < 0:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.min(input).item()),
            weight_size[1],
        )
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.max(input).item()),
            weight_size[1],
        )
    if offsets is not None and len(input.size()) != 1:
        raise ValueError("Input dimension needs to be exactly 1 dim")
    if len(input.size()) == 1 and offsets is None:
        raise ValueError("offsets is required for 1D input")
    if per_sample_weights is not None and per_sample_weights.size() != input.size():
        raise ValueError(
            f"per_sample_weights size {per_sample_weights.size()} not equal to input size {input.size()}"
        )
    if mode is None:
        mode = "mean"
    if mode not in ["sum", "mean", "max"]:
        raise ValueError(f"mode '{mode}' is not supported")
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )
    if include_last_offset and offsets is None:
        raise ValueError('offsets is required for flag "include_last_offset"!')
    if include_last_offset and cast(list[int], offsets)[-1] != input.size(0):
        raise ValueError(
            'offsets need to have the input size in the end when the flag "include_last_offset" is on!'
        )

    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(
    input,
    world_size,
    weight,
    local_shard,
    offsets,
    per_sample_weights,
    mode,
    max_norm,
    norm_type,
    padding_idx,
    pg,
):
    # [FUNCTION DOC] HELPER: Column-wise path. Gathers ids, per-sample weights
    # and offsets across pg, optionally renorms the local shard in place when
    # max_norm is given, then runs the shared column-wise base routine with
    # functional embedding_bag. Returns a tuple of (output, local shard, which is
    # the renormed shard when max_norm was used).
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        pg: process group.

    Return:
        output: final result of lookup and aggregation.
        local_shard: col-wise shared local weight used for lookup.
            If max_norm, this will be the renormed weight.
    """
    # allgather the special input of embedding bag first.
    (
        gathered_inputs,
        gathered_per_sample_weights,
        gathered_offsets,
    ) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)

    if max_norm is not None:
        # max_norm changes the weight in-place
        local_shard = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg
        )

    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding_bag,
        1,
        input,
        world_size,
        weight,
        local_shard,
        pg,
        gathered_inputs,
        mode=mode,
        gathered_per_sample_weights=gathered_per_sample_weights,
        gathered_offsets=gathered_offsets,
        padding_idx=padding_idx,
    )
    return (output, local_shard)


def _handle_row_wise_sharding(
    input,
    world_size,
    weight,
    local_shard,
    offsets,
    per_sample_weights,
    mode,
    max_norm,
    norm_type,
    padding_idx,
    rank,
    pg,
):
    # [FUNCTION DOC] HELPER: Row-wise path. Gathers ids (plus per-sample weights
    # and offsets when present, concatenating them and shifting each rank's
    # offsets), masks ids not owned by this rank to an extra padding row (filled
    # with negative infinity in max mode), optionally applies the max_norm renorm
    # workaround for GH issue 81717, runs embedding_bag against local shard plus
    # padding row (mean is computed as sum), then reduce-scatters across pg with
    # sum, or max for max mode. For mean mode divides by bag sizes derived from
    # the input (non-padding counts for 2D input, offset differences for 1D).
    # Returns this rank's output.
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        rank: # of cuda process.
        pg: process group.

    Returns:
        gathered_output: final result of lookup and aggregation.
    """
    if input.dim() > 1 and per_sample_weights is None:
        # allgather the inputs first for non Replicated Tensor.
        gather_inp = _all_gather_base_input(input, pg)
    else:
        (
            gathered_inputs,
            gathered_per_sample_weights,
            gathered_offsets,
        ) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)
        cat_dim = 0 if input.dim() != 1 else -1
        gather_inp = torch.cat(gathered_inputs, dim=cat_dim)
        if per_sample_weights is not None:
            per_sample_weights = torch.cat(gathered_per_sample_weights, dim=cat_dim)
        offset_add = 0 if input.dim() > 1 else input.size(0)
        if offsets is not None:
            offsets_list = torch.cat(
                [gathered_offsets[i] + (offset_add * i) for i in range(pg.size())],
                dim=cat_dim,
            )

    # Mask the input according to sharding spec.
    lookup_input, padding_local, padding_row = _handle_row_wise_mask(
        gather_inp, padding_idx, weight, world_size, rank
    )
    if mode == "max":
        padding_row[:] = -float("Inf")

    # When input is a large tensor, the value of weight is changed.
    # This is a walk-around for now. GH issue: #81717.
    if max_norm is not None:
        torch.nn.functional.embedding_bag(
            torch.unique(lookup_input)[:-1],
            local_shard,
            offsets=torch.tensor([0], device=local_shard.device, dtype=torch.long),
            mode=mode,
            per_sample_weights=None,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_local,
        )
        max_norm = None
    result = torch.nn.functional.embedding_bag(
        lookup_input,
        torch.cat([local_shard, padding_row]),
        offsets=offsets_list if offsets is not None else offsets,  # type: ignore[possibly-undefined]
        mode=mode if mode != "mean" else "sum",
        per_sample_weights=per_sample_weights,
        max_norm=max_norm,
        norm_type=norm_type,
        padding_idx=padding_local,
    )

    op = ReduceOp.SUM if mode != "max" else ReduceOp.MAX
    # TODO: Make the result a PartialTensor and move the logic below there.
    local_shards = result.chunk(pg.size())
    result = reduce_scatter(
        torch.empty_like(local_shards[0]),
        list(local_shards),
        op=op,
        group=pg,
    )

    # For Mean, we cannot do the division until very end because the sum of means
    # not equal to the mean of sum. (Divisor is different)
    if mode == "mean":
        if input.dim() > 1:
            padding_idx = padding_idx if padding_idx is not None else -1
            split_sizes = torch.sum(
                torch.ne(input, padding_idx), dim=-1, dtype=local_shard.dtype
            )
        else:
            split_sizes = torch.cat(
                (
                    offsets[1 : offsets.size(0)] - offsets[0:-1],
                    (input.size(0) - offsets[-1]).unsqueeze(0),
                ),
                dim=-1,
            )
        return torch.div(result, split_sizes.unsqueeze(1))

    # Return the appropriate local result.
    return result


def _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg):
    # [FUNCTION DOC] HELPER: Stacks input, optional per_sample_weights and
    # optional offsets (resized to the input's shape) into one tensor and
    # all-gathers it once over pg. Returns three values: the list of per-rank
    # inputs (original dtype), the list of per-rank per_sample_weights or None,
    # and the list of per-rank offsets (original size and dtype) or None.
    # Note the stack mixes dtypes, so ids and offsets pass through the common
    # promoted dtype before being cast back.
    """
    In case we need to gather input and all other parameters of embeddingBag
    ops, we need to stack all input together to perform ``all_gather``
    collective communication just once.

    Note that since offsets does not share the same size as input and
    is always smaller than input, we resize it during the communication.

    Args:
        input: tensor to be applied op on.
        per_sample_weights: weights for weighted sum mode.
        offsets: when input is 1D. offsets determines the starting
            index position of each bag (sequence) in input.
        pg: process group.

    Returns:
        gathered_inputs: list of input tensor gathered from each rank.
        gathered_per_sample_weights: list of per_sample_weights from each rank.
        gathered_offsets: list of offsets from each rank.
    """
    input_to_gather = [input]
    if per_sample_weights is not None:
        input_to_gather.append(per_sample_weights)
    if offsets is not None:
        input_to_gather.append(offsets.clone().resize_(input.size()))
    gathered_inputs = all_gather(torch.stack(input_to_gather), group=pg)

    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        gathered_per_sample_weights = [t[1] for t in gathered_inputs]
    gathered_offsets = None
    if offsets is not None:
        idx = 2 if per_sample_weights is not None else 1
        gathered_offsets = [
            t[idx].resize_(offsets.size()).to(offsets.dtype) for t in gathered_inputs
        ]
    gathered_inputs = [t[0].to(input.dtype) for t in gathered_inputs]
    return gathered_inputs, gathered_per_sample_weights, gathered_offsets
