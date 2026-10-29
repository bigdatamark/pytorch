# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC]
# Registers DTensor single-dim sharding strategies for the embedding forward op
# and the embedding dense backward op. Each strategy is a list of placement lists
# (output first, then inputs). Import-time side effect: the two strategy
# functions are registered with the single-dim strategy registry.
import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    register_single_dim_strategy,
)
from torch.distributed.tensor.placement_types import (
    _MaskPartial,
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


@register_single_dim_strategy(aten.embedding.default)
def embedding_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement]]:
    # [FUNCTION DOC] Builds candidate placement lists for embedding, ordered
    # [output, weight, indices]: colwise (output and weight sharded on the
    # embedding dim, indices replicated), rowwise (weight sharded on dim 0, output
    # and indices share one mask-partial placement object so the mask buffer is
    # shared), and one batch-dim strategy per indices dim (weight replicated).
    # Weight shape is concretized to ints so it is hashable under tracing.
    # Raises AssertionError if weight or indices metadata is not tensor metadata.
    """Single-dim strategy for embedding: rowwise, colwise, and batch-dim sharding.

    Placement order: [output, weight, indices]
    """
    weight_meta = args_schema[0]
    indices_meta = args_schema[1]
    if not isinstance(weight_meta, TensorMeta) or not isinstance(
        indices_meta, TensorMeta
    ):
        raise AssertionError

    # _MaskPartial hashes offset_shape, but torch.Size with SymInt (from
    # dynamo tracing) is unhashable.  Concretize to int, which adds a
    # standard dynamo guard (recompiles if the shape changes at runtime).
    weight_shape = torch.Size(int(s) for s in weight_meta.shape)
    indices_shape = indices_meta.shape
    output_emb_dim = len(indices_shape)

    strategies: list[list[Placement]] = []

    # colwise: output shard on last dim, weight shard on dim 1, indices replicate
    strategies.append([Shard(output_emb_dim), Shard(1), Replicate()])

    # rowwise: output is MaskPartial, weight shard on dim 0, indices MaskPartial
    # NOTE: same object for output & indices so the mask buffer is shared
    embedding_partial = _MaskPartial(offset_shape=weight_shape, offset_dim=0)
    strategies.append([embedding_partial, Shard(0), embedding_partial])

    # batch dim sharding: weight replicated, indices shard on any dim, output follows
    for i in range(len(indices_shape)):
        strategies.append([Shard(i), Replicate(), Shard(i)])

    return strategies


@register_single_dim_strategy(
    aten.embedding_dense_backward.default,
    schema_info=RuntimeSchemaInfo(static_argnum=2),
)
def embedding_dense_backward_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement]]:
    # [FUNCTION DOC] Builds candidate placement lists for embedding backward,
    # ordered [weight_grad, grad_out, indices]: colwise (weight grad sharded on
    # dim 1, grad_out sharded on its last dim, indices replicated), one batch-dim
    # strategy per indices dim (weight grad partial, grad_out and indices sharded
    # on that dim), and a fully partial grad_out with replicated indices.
    # Raises AssertionError if grad_out or indices metadata is not tensor metadata.
    """Single-dim strategy for embedding backward.

    Placement order: [output(weight_grad), grad_out, indices]
    """
    grad_out_meta = args_schema[0]
    indices_meta = args_schema[1]
    if not isinstance(grad_out_meta, TensorMeta) or not isinstance(
        indices_meta, TensorMeta
    ):
        raise AssertionError

    grad_out_ndim = len(grad_out_meta.shape)
    indices_shape = indices_meta.shape

    strategies: list[list[Placement]] = []

    # colwise backward: weight grad shard on dim 1, grad_out shard on last dim, indices replicate
    strategies.append([Shard(1), Shard(grad_out_ndim - 1), Replicate()])

    # batch dim sharding: weight grad partial, grad_out/indices shard on same dim
    for i in range(len(indices_shape)):
        strategies.append([Partial(), Shard(i), Shard(i)])

    # grad_out partial, indices replicate, weight grad partial
    strategies.append([Partial(), Partial(), Replicate()])

    return strategies
