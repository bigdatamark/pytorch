# mypy: allow-untyped-defs
# [MODULE DOC] Microbatch splitting and merging utilities for pipeline parallelism.
# Provides TensorChunkSpec (per-argument chunk dimension), split_args_kwargs_into_chunks
# (split nested args/kwargs into microbatches, replicating non-tensors by default) and
# merge_chunks (inverse: concatenate, reduce or verify-equal per-microbatch outputs).
# Supports plain tensors, DTensors and flex-attention BlockMasks. Module-level state: a
# debug flag _debug_mask_minibatches (default off), DEFAULT_CHUNK_DIM = 0 and a
# sum_reducer for losses. No import-time side effects besides creating a logger and tensor.
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch.distributed.tensor import DTensor
from torch.fx.node import map_aggregate
from torch.nn.attention.flex_attention import BlockMask
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


__all__ = [
    "TensorChunkSpec",
    "split_args_kwargs_into_chunks",
    "merge_chunks",
]

logger = logging.getLogger(__name__)

"""
_debug_mask_minibatches specifies to send masked versions of the mini-batch
through instead of micro-batch slices--this can be used for more stable
numerical testing (see [A Note About Correctness Testing])
"""
_debug_mask_minibatches = False


class _CustomReducer:
    # [CLASS DOC] HELPER: Chunk spec for outputs reduced across microbatches. Attributes:
    # init_value (starting accumulator) and reduce_fn (binary function folding each chunk in).
    """
    Custom reducer class that can be used to specify a custom operation that
    reduces losses of multiple microbatches into one value.

    Example:
    >>> # xdoctest: +SKIP
    >>> sum_reducer = _CustomReducer(
    >>>     torch.tensor(0.0),
    >>>     lambda a, b: a + b
    >>> )
    """

    def __init__(self, init_value, reduce_fn):
        # [FUNCTION DOC] HELPER: Stores the initial accumulator value and the binary reduce function.
        self.init_value = init_value
        self.reduce_fn = reduce_fn


class _LossReducer(_CustomReducer):
    # [CLASS DOC] HELPER: Reducer subclass used as a chunk spec for loss outputs; adds no behavior.
    pass


sum_reducer = _LossReducer(torch.tensor(0.0), operator.add)

# Default chunking dimension is 0. This is used for the case where the user did
# not specify a chunking dimension.
DEFAULT_CHUNK_DIM = 0


class TensorChunkSpec:
    # [CLASS DOC] Spec for one tensor argument: holds split_dim, the dimension to chunk along
    # (and to concatenate along when merging). Build trees of these with from_tuple/from_dict.
    """
    Class used to specify chunking of inputs
    """

    def __init__(self, split_dim):
        # [FUNCTION DOC] Stores the dimension along which a tensor is split into microbatches.
        self.split_dim = split_dim

    split_dim: int

    def __repr__(self):
        # [FUNCTION DOC] Representation including module-qualified class name and split dimension.
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}({self.split_dim})"
        )

    def __str__(self):
        # [FUNCTION DOC] Short string form, class name and split dimension only.
        return f"TensorChunkSpec({self.split_dim})"

    @staticmethod
    def from_tuple(
        chunk_dims: tuple[int, ...],
    ):
        # [FUNCTION DOC] Static. Maps a tuple of chunk dimensions to a same-shaped tuple of
        # TensorChunkSpec objects.
        """
        A helper for creating a tuple of `TensorChunkSpec` from a tuple of chunk
        dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # There are three positional arguments to the model, and
            >>> # we are chunking them along dimension 0, 0 and 1, respectively
            >>> args_chunk_spec = TensorChunkSpec.from_tuple((0, 0, 1))
        """
        args_chunk_spec = map_aggregate(
            chunk_dims,
            lambda dim: TensorChunkSpec(dim),  # type: ignore[arg-type,return-value]
        )
        return args_chunk_spec

    @staticmethod
    def from_dict(
        chunk_dims: dict[str, int],
    ):
        # [FUNCTION DOC] Static. Maps a dict of chunk dimensions to a same-shaped dict of
        # TensorChunkSpec objects.
        """
        A helper for creating a dictionary of `TensorChunkSpec` from a
        dictionary of chunk dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # Chunk dimension 0 for the "id" argument, 1 for the "mask" argument
            >>> kwargs_chunk_spec = TensorChunkSpec.from_dict({"id": 0, "mask": 1})
        """
        kwargs_chunk_spec = map_aggregate(
            chunk_dims,
            lambda dim: TensorChunkSpec(dim),  # type: ignore[arg-type,return-value]
        )
        return kwargs_chunk_spec


# Class used to specify replication of inputs
class _Replicate:
    # [CLASS DOC] HELPER: Marker spec meaning the value is copied unchanged to every chunk.
    pass


def _split_block_mask(
    block_mask: BlockMask,
    num_chunks: int,
) -> list[BlockMask]:
    # [FUNCTION DOC] HELPER: Splits a BlockMask along the batch dimension into num_chunks masks.
    # A mask with batch size 1 is broadcast (same object repeated). Otherwise the block tensors
    # (including optional full-block ones) are split and each chunk's mask_mod is wrapped to add
    # the chunk's batch offset. Raises AssertionError if batch size is below num_chunks.
    """Given a block mask, split the block mask along the batch dimension (dim0).

    Args:
        block_mask: Block mask to split
        num_chunks: Number of chunks to split the block mask into

    Returns:
        chunk_block_masks: List of chunked block masks
    """

    # BlockMask will broadcast if B is 1.
    if block_mask.kv_num_blocks.size(0) == 1:
        return [block_mask] * num_chunks

    if not block_mask.kv_num_blocks.size(0) >= num_chunks:
        raise AssertionError(
            "Block mask has fewer batch size than the number of chunks. "
        )

    batch_dim = 0
    kv_num_blocks_chunks = torch.tensor_split(
        block_mask.kv_num_blocks, num_chunks, batch_dim
    )
    kv_indices_chunks = torch.tensor_split(block_mask.kv_indices, num_chunks, batch_dim)
    full_kv_num_blocks_chunks = (
        torch.tensor_split(block_mask.full_kv_num_blocks, num_chunks, batch_dim)
        if block_mask.full_kv_num_blocks is not None
        else [None] * num_chunks
    )
    full_kv_indices_chunks = (
        torch.tensor_split(block_mask.full_kv_indices, num_chunks, batch_dim)
        if block_mask.full_kv_indices is not None
        else [None] * num_chunks
    )

    chunk_block_masks = []
    batch_offset = 0
    for chunk_idx in range(num_chunks):

        def create_mask_mod(idx):
            # [FUNCTION DOC] HELPER: Closure factory capturing the batch offset idx for one chunk;
            # returns a mask function that shifts batch indices by idx.
            def batch_offset_mask_mod(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: Calls the original mask_mod with the batch index shifted by
                # the chunk's offset so a chunk sees its global batch positions.
                b_offset = torch.full_like(b, idx)
                return block_mask.mask_mod(b + b_offset, h, q_idx, kv_idx)

            return batch_offset_mask_mod

        chunk_block_masks.append(
            BlockMask.from_kv_blocks(
                kv_num_blocks=kv_num_blocks_chunks[chunk_idx],
                kv_indices=kv_indices_chunks[chunk_idx],
                full_kv_num_blocks=full_kv_num_blocks_chunks[chunk_idx],
                full_kv_indices=full_kv_indices_chunks[chunk_idx],
                BLOCK_SIZE=block_mask.BLOCK_SIZE,
                mask_mod=create_mask_mod(batch_offset),
                seq_lengths=block_mask.seq_lengths,
            )
        )
        batch_offset += kv_num_blocks_chunks[chunk_idx].size(0)
    return chunk_block_masks


def _split_tensor(
    tensor: torch.Tensor,
    spec: TensorChunkSpec,
    num_chunks: int,
) -> Sequence[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Splits one tensor into num_chunks pieces along spec.split_dim
    # (uneven split via tensor_split). Raises AssertionError if the dimension is smaller than
    # num_chunks. DTensors are split locally and each piece rewrapped with the correct global
    # shape and the original placements. Leaf tensors requiring grad get retain_grad on each
    # chunk so backward can read gradients. In debug-mask mode returns full-size zero-padded
    # tensors instead of slices.
    """Given a tensor, and a chunking spec, split the tensor.
    Args:

        tensor: Tensor to split
        spec: Chunking spec
        num_chunks: Number of chunks to split the tensor into

    Returns:
        chunk_tensors: List of chunked tensors
    """

    if not tensor.size(spec.split_dim) >= num_chunks:
        raise AssertionError(
            f"Tensor size {tensor.size(spec.split_dim)} is smaller than num_chunks"
        )

    _is_dtensor = isinstance(tensor, DTensor)

    if _is_dtensor:
        # Split locally and wrap each chunk as a DTensor with the correct
        # global shape.  We avoid DTensor dispatch (which would all-gather
        # Shard(split_dim) to Replicate) and avoid local_map (whose
        # from_local call assumes even sharding and computes wrong global
        # sizes when local shards differ across ranks).
        placements = tensor.placements
        mesh = tensor.device_mesh
        local_tensor = tensor.to_local()
        local_chunks = torch.tensor_split(local_tensor, num_chunks, spec.split_dim)
        # Compute the correct global chunk sizes along split_dim.
        global_split_size = tensor.shape[spec.split_dim]
        quotient, remainder = divmod(global_split_size, num_chunks)
        global_stride = tensor.stride()
        chunk_tensors_list: list[torch.Tensor] = []
        for i, local_chunk in enumerate(local_chunks):
            chunk_shape = list(tensor.shape)
            chunk_shape[spec.split_dim] = quotient + (1 if i < remainder else 0)
            chunk_tensors_list.append(
                DTensor.from_local(
                    local_chunk,
                    mesh,
                    placements,
                    shape=torch.Size(chunk_shape),
                    stride=global_stride,
                    run_check=False,
                )
            )
        chunk_tensors: Sequence[torch.Tensor] = chunk_tensors_list  # type: ignore[assignment]
    else:
        chunk_tensors = torch.tensor_split(tensor, num_chunks, spec.split_dim)

    # tensor_split on a leaf tensor produces non-leaf views that won't
    # accumulate .grad during torch.autograd.backward().  Call retain_grad()
    # on those views so that stage_backward() can read .grad from them.
    if tensor.requires_grad and tensor.is_leaf:
        for chunk in chunk_tensors:
            chunk.retain_grad()

    if not _debug_mask_minibatches:
        return chunk_tensors

    def _expand_chunks(
        orig: torch.Tensor, *chunks: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        # [FUNCTION DOC] HELPER: Debug mode only. For each chunk, builds a zero tensor shaped like
        # orig and copies the chunk into its contiguous slot along split_dim, giving full-size
        # masked mini-batches. Returns a tuple of these tensors.
        expanded = []
        idx = 0
        for chunk in chunks:
            new_val = torch.zeros_like(orig)
            upper = idx + chunk.size(spec.split_dim)
            slices: list[slice] = [slice(None)] * new_val.ndim
            slices[spec.split_dim] = slice(idx, upper)
            new_val[slices] = chunk
            expanded.append(new_val)
            idx += chunk.size(spec.split_dim)
        return tuple(expanded)

    if _is_dtensor:
        placements = tensor.placements
        mesh = tensor.device_mesh
        global_shape = tensor.shape
        global_stride = tensor.stride()
        local_expanded = _expand_chunks(
            tensor.to_local(),
            *(cast(DTensor, c).to_local() for c in chunk_tensors),
        )
        return [
            DTensor.from_local(
                t,
                mesh,
                placements,
                shape=global_shape,
                stride=global_stride,
                run_check=False,
            )
            for t in local_expanded
        ]
    else:
        return list(_expand_chunks(tensor, *chunk_tensors))


def _shard_dict_of_args(
    args_dict,
    args_chunk_spec,
    num_chunks,
):
    # [FUNCTION DOC] HELPER: Shards a dict (or index-keyed dict) of args per a parallel spec
    # structure. Flattens both (BlockMask treated as a leaf), computes the usable chunk count
    # as the minimum of num_chunks and each split dimension size, then splits tensors, splits
    # BlockMasks, or replicates values marked _Replicate, and unflattens one structure per
    # chunk. Empty input gives num_chunks empty dicts. Raises AssertionError on size/spec
    # mismatch or wrong spec type, ValueError for unsupported value/spec combinations.
    """
    Given a dictionary of args, and a dictionary of chunking specs, shard the
    args according to the chunking specs.

    Args:
        args_dict: Dictionary of args
        args_chunk_spec: Dictionary of chunking specs
        num_chunks: Number of chunks to shard the args into

    Returns:
        args_split: List of sharded args
    """

    if not args_dict:
        return [{} for _ in range(num_chunks)]

    if not len(args_dict) == len(args_chunk_spec):
        raise AssertionError(
            f"args_dict.keys() = {list(args_dict.keys())} "
            f"args_chunk_spec.keys() = {list(args_chunk_spec.keys())}"
        )
    if args_chunk_spec is None:
        raise AssertionError("args_chunk_spec should have been set by caller")

    values, tree_spec = tree_flatten(
        args_dict, is_leaf=lambda x: isinstance(x, BlockMask)
    )
    chunk_specs, _ = tree_flatten(
        args_chunk_spec, is_leaf=lambda x: isinstance(x, BlockMask)
    )

    # First check and find the actual number of chunks
    split_sizes = []
    for v, spec in zip(values, chunk_specs, strict=True):
        # The original logic is "spec is _Replicate". This doesn't seem to be
        # correct. But we keep it for backward compatibility.
        if spec is _Replicate or isinstance(spec, _Replicate):
            split_sizes.append(num_chunks)
        elif isinstance(v, torch.Tensor):
            if not isinstance(spec, TensorChunkSpec):
                raise AssertionError(f"Expected TensorChunkSpec, got {type(spec)}")
            split_sizes.append(v.size(spec.split_dim))
        elif isinstance(v, BlockMask):
            if not isinstance(spec, TensorChunkSpec):
                raise AssertionError(f"Expected TensorChunkSpec, got {type(spec)}")
            if not spec.split_dim == 0:
                raise AssertionError("BlockMask only supports split_dim=0")
            # BlockMask will broadcast if B is 1.
            if v.kv_num_blocks.size(0) == 1:
                split_sizes.append(num_chunks)
            else:
                split_sizes.append(v.kv_num_blocks.size(0))
        else:
            raise ValueError(
                f"Unsupported chunk spec: {spec} and value: {v} combination."
            )
    result_num_chunks = min(*split_sizes, num_chunks)

    flat_split_results: list[Any] = [[] for _ in range(result_num_chunks)]
    for v, spec in zip(values, chunk_specs, strict=True):
        v_splits: Sequence[Any] = []
        if spec is _Replicate or isinstance(spec, _Replicate):
            v_splits = [v] * result_num_chunks
        elif isinstance(v, torch.Tensor):
            v_splits = _split_tensor(v, spec, result_num_chunks)
        elif isinstance(v, BlockMask):
            v_splits = _split_block_mask(v, result_num_chunks)
        else:
            raise ValueError(
                f"Unsupported chunk spec: {spec} and value: {v} combination."
            )

        for _flat_split_result, _v_split in zip(
            flat_split_results, v_splits, strict=True
        ):
            _flat_split_result.append(_v_split)

    return [
        tree_unflatten(_flat_split_result, tree_spec)
        for _flat_split_result in flat_split_results
    ]


def split_args_kwargs_into_chunks(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    chunks: int,
    args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
    kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
) -> tuple[list[tuple], list[dict]]:
    # [FUNCTION DOC] Splits positional args and keyword args into microbatches. None kwargs
    # become empty; missing specs default to splitting tensors/BlockMasks on dim 0 and
    # replicating everything else. Args are sharded first, kwargs are sharded to that count,
    # and if kwargs yield fewer chunks args are re-sharded to match. The actual chunk count
    # may be smaller than requested when a tensor is too small. Returns (list of arg tuples,
    # list of kwarg dicts), equal length. Raises RuntimeError if counts still differ.
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to  their respective chunking specs.

    Args:
        args: Tuple of args
        kwargs: Dict of kwargs
        chunks: Number of chunks to split the args and kwargs into
        args_chunk_spec: chunking specs for args, in same shape as args
        kwargs_chunk_spec: chunking specs for kwargs, in same shape as kwargs

    Returns:
        args_split: List of sharded args
        kwargs_split: List of sharded kwargs
    """
    # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
    # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
    # and `kwargs_chunk_spec` specifications. The steps are as follows:
    #
    # 1. Use pytree.tree_flatten to flatten each arg and its spec into a 1d array of values.
    #    To use a running example: suppose our inputs look like
    #
    #       args = ([A, [B, C]], D) args_spec = ([None, [None, TensorChunkSpec]], None)
    #       (kwargs not shown but it's a similar process)
    #
    #    Then for this step we would end up with
    #
    #       args = ([A, B, C], D) args_spec = ([None, None, TensorChunkSpec], None)
    #
    # 2. Shard or replicate the arguments subject to the policy in the spec. Suppose chunks = 2
    #
    #       args = ([[A, A], [B, B], [C_1, C_2]], [D, D])
    #
    # 3. Rotate the nesting order such that chunks are the outer dimension
    #
    #       args_chunks = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 4. Unflatten each chunk according to the spec
    #
    #       args_chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]

    # TODO: _debug_mask_minibatches
    # Handle the case where kwargs is None
    if kwargs is None:
        kwargs = {}

    # If user did not provide args_chunk_spec or kwargs_chunk_spec, we extend
    # their format and use default chunking along dim 0
    def default_spec(v):
        # [FUNCTION DOC] HELPER: Default chunk spec for one leaf: TensorChunkSpec on the default
        # dim for tensors and BlockMasks, otherwise a replicate marker.
        if isinstance(v, torch.Tensor | BlockMask):
            return TensorChunkSpec(DEFAULT_CHUNK_DIM)
        else:
            return _Replicate()

    if args_chunk_spec is None:
        args_chunk_spec = tree_map(
            default_spec, args, is_leaf=lambda v: isinstance(v, BlockMask)
        )

    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = tree_map(
            default_spec, kwargs, is_leaf=lambda v: isinstance(v, BlockMask)
        )

    args_split_dict = _shard_dict_of_args(
        dict(enumerate(args)),
        dict(enumerate(args_chunk_spec)),
        chunks,
    )
    real_num_chunks = len(args_split_dict)

    kwargs_split = _shard_dict_of_args(
        kwargs,
        kwargs_chunk_spec,
        real_num_chunks,
    )

    if len(kwargs_split) < real_num_chunks:
        # In case kwargs are sharded into less chunks
        # e.g. when `args` has no tensor, just values
        real_num_chunks = len(kwargs_split)
        # Re-shard args
        args_split_dict = _shard_dict_of_args(
            dict(enumerate(args)),
            dict(enumerate(args_chunk_spec)),
            real_num_chunks,
        )

    if len(args_split_dict) != len(kwargs_split):
        raise RuntimeError(
            "args and kwargs are split into different number of chunks: "
            f"{len(args_split_dict)}, {len(kwargs_split)}"
        )

    args_split = [
        tuple(chunk_args[i] for i in range(len(chunk_args)))
        for chunk_args in args_split_dict
    ]

    return args_split, kwargs_split


def merge_chunks(
    chunks: list[Any],
    chunk_spec,
):
    # [FUNCTION DOC] Inverse of split_args_kwargs_into_chunks. Takes a list of per-microbatch
    # outputs and a chunk spec tree (or None, meaning concatenate every leaf along dim 0).
    # Leaves with a TensorChunkSpec are concatenated along split_dim (DTensors are
    # concatenated locally and rewrapped with the summed global shape; mixed or
    # placement-mismatched DTensors raise AssertionError); _CustomReducer leaves are folded
    # with reduce_fn from init_value; other leaves must be equal across chunks (else
    # AssertionError). Raises ValueError if a chunk's leaf count differs from the spec.
    # In debug-mask mode each chunk is first sliced back to its own region. Returns merged tree.
    """
    Given a list of chunks, merge them into a single value according to
    the chunk spec.

    Args:
        chunks: list of chunks
        chunk_spec: Chunking spec for the chunks

    Returns:
        value: Merged value
    """
    # This is essentially the inverse of `split_args_kwargs_into_chunks`, so the
    # steps are similar to the steps in that function but in reverse. Given the
    # input values:
    #
    #       chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]
    #       args_spec = ([None, [None, TensorChunkSpec]], None)
    #
    # 1. Flatten the chunks according to the chunk_spec
    #
    #       chunks_flat = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 2. Rotate the nesting order such that chunks are the inner dimension
    #
    #       value_inner = ([A, B, [C_1, C_2]], D)
    #
    # 3. Concatenate sharded arguments
    #
    #       value_combined = ([A, B, C], D)
    #
    # 4. Unflatten the combined args given the spec
    #
    #       value = ([A, [B, C]], D)

    # Preliminary: flatten the chunk spec
    if chunk_spec is not None:
        spec_flattened, flatten_spec = tree_flatten(chunk_spec)
    else:
        # If chunk_spec is not provided, we will merge chunks along the default dimension (0), for all output fields
        # We obtain the output structure by flattening chunk 0 and generate the chunk_spec
        chunk0_flat, flatten_spec = tree_flatten(chunks[0])
        spec_flattened = [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(chunk0_flat)

    # Stage 1: flatten chunks
    # chunks_flattened : [num chunks, num args]
    chunks_flattened = []

    for chunk in chunks:
        chunk_flattened, _ = tree_flatten(chunk)
        if len(chunk_flattened) != len(spec_flattened):
            raise ValueError(f"Chunk {chunk} did not match chunk spec {chunk_spec}")

        chunks_flattened.append(chunk_flattened)

    # Stage 2 and 3: Rotate nesting order s.t. chunks are inner dimension and
    #                concatenate sharded operands
    # args_flattened : [num args]
    args_flattened: list[Any] = []
    for arg_idx, arg in enumerate(spec_flattened):
        if isinstance(arg, TensorChunkSpec):
            partial_values = [
                chunks_flattened[chunk_idx][arg_idx]
                for chunk_idx in range(len(chunks_flattened))
            ]

            if _debug_mask_minibatches:
                # Infer size of individual chunks by running `tensor_split` again
                overall_shape = partial_values[0].shape
                for val in partial_values[1:]:
                    if not val.shape == overall_shape:
                        raise AssertionError(
                            f"Expected shape {overall_shape}, got {val.shape}"
                        )
                meta_chunks = torch.tensor_split(
                    torch.empty(*overall_shape, device="meta"),
                    sections=len(partial_values),
                    dim=arg.split_dim,
                )

                values_to_cat = []
                chunk_start_idx = 0
                if not len(partial_values) == len(meta_chunks):
                    raise AssertionError(
                        f"Expected len(partial_values) == len(meta_chunks), got {len(partial_values)} != {len(meta_chunks)}"
                    )

                for partial_value, meta_chunk in zip(
                    partial_values, meta_chunks, strict=True
                ):
                    chunk_end_idx = chunk_start_idx + meta_chunk.size(arg.split_dim)

                    slice_indices = [slice(None, None, None)] * partial_value.ndim
                    slice_indices[arg.split_dim] = slice(chunk_start_idx, chunk_end_idx)
                    sliced = partial_value[slice_indices]
                    values_to_cat.append(sliced)

                    chunk_start_idx = chunk_end_idx

            else:
                values_to_cat = partial_values

            # Validate DTensor consistency: either all values are DTensors
            # or none are. A mix indicates a bug in the pipeline stage.
            dtensor_flags = [isinstance(v, DTensor) for v in values_to_cat]
            if any(dtensor_flags):
                if not all(dtensor_flags):
                    raise AssertionError(
                        "merge_chunks: expected all values to be DTensors or "
                        "none to be DTensors, got a mix"
                    )
                # All DTensors must have matching placements.
                placements = values_to_cat[0].placements
                for i, v in enumerate(values_to_cat[1:], 1):
                    if v.placements != placements:
                        raise AssertionError(
                            f"merge_chunks: placement mismatch at chunk {i}: "
                            f"expected {placements}, got {v.placements}"
                        )
                mesh = values_to_cat[0].device_mesh
                local_cat = torch.cat(
                    [v.to_local() for v in values_to_cat],
                    dim=arg.split_dim,
                )
                cat_shape = list(values_to_cat[0].shape)
                cat_shape[arg.split_dim] = sum(
                    v.shape[arg.split_dim] for v in values_to_cat
                )
                cat_shape = torch.Size(cat_shape)
                cat_stride = values_to_cat[0].stride()
                args_flattened.append(
                    DTensor.from_local(
                        local_cat,
                        mesh,
                        placements,
                        shape=cat_shape,
                        stride=cat_stride,
                        run_check=False,
                    )
                )
            else:
                args_flattened.append(torch.cat(values_to_cat, dim=arg.split_dim))
        elif isinstance(arg, _CustomReducer):
            reduced_val = arg.init_value

            for chunk_idx in range(len(chunks_flattened)):
                reduced_val = arg.reduce_fn(
                    reduced_val, chunks_flattened[chunk_idx][arg_idx]
                )

            args_flattened.append(reduced_val)
        else:
            value = chunks_flattened[0][arg_idx]
            for chunk_idx in range(1, len(chunks_flattened)):
                if not chunks_flattened[chunk_idx][arg_idx] == value:
                    raise AssertionError(
                        f"Expected {value}, got {chunks_flattened[chunk_idx][arg_idx]}"
                    )
            args_flattened.append(value)

    # Stage 4: Unflatten combined args
    return tree_unflatten(args_flattened, flatten_spec)
