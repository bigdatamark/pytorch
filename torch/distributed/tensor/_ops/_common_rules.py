# Copyright (c) Meta Platforms, Inc. and affiliates
# [MODULE DOC]
# Shared sharding-propagation rules for DTensor ops expressed in einsum
# notation. Provides einop_rule (generic einsum-style propagation producing an
# OutputSharding, either an output spec or a reshard suggestion) and
# pointwise_rule (builds an einsum equation with broadcasting support and
# delegates to einop_rule), plus small private helpers. No import-time side effects.
import string
from typing import cast

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import prod
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


def _replace_char_in_str(string: str, new_char: str, idx: int) -> str:
    # [FUNCTION DOC] HELPER: Returns a copy of the string with the character at
    # position idx replaced by new_char.
    return string[:idx] + new_char + string[idx + 1 :]


def _gen_reshard_suggestions(
    op_schema: OpSchema,
    input_dims: list[str],
    input_specs: tuple[DTensorSpec, ...],
    dim_to_sharding: dict[str, int],
    pending_sum: list[int],
) -> OutputSharding:
    # [FUNCTION DOC] HELPER: Builds an OutputSharding with no output spec but a
    # redistribute_schema. For each input, a new DTensorSpec is created on the
    # same mesh from a dim map derived from dim_to_sharding (mesh dim per einsum
    # letter), with pending_sum mesh dims as partial sums and the original
    # tensor_meta. The resulting schema keeps the op and copies kwargs/rewrap
    # info from op_schema via _inplace_rewrap_schema_suggestion.
    suggested_arg_specs: list[DTensorSpec] = []
    for input_dim, input_spec in zip(input_dims, input_specs):
        dim_map = [dim_to_sharding[dim] for dim in input_dim]
        suggested_arg_specs.append(
            DTensorSpec.from_dim_map(
                mesh=input_spec.mesh,
                dim_map=dim_map,
                sums=pending_sum,
                tensor_meta=input_spec.tensor_meta,
            )
        )
    suggested_schema = OpSchema(op_schema.op, tuple(suggested_arg_specs), {})
    suggested_schema._inplace_rewrap_schema_suggestion(op_schema)
    return OutputSharding(
        None,
        redistribute_schema=suggested_schema,
    )


def einop_rule(
    equation: str,
    op_schema: OpSchema,
    *,
    linearity: bool = False,
    enforce_sharding: dict[str, int] | None = None,
) -> OutputSharding:
    # [FUNCTION DOC]
    # Propagates input DTensor shardings through an einsum-style equation (single
    # output). Records the mesh dim sharding and size of each dim letter, merging
    # across inputs (replicate merges into shard but flags a reshard; two
    # different shardings raise RuntimeError). Flags a reshard when two letters
    # land on the same mesh dim (keeping the one with largest local-size cost and
    # replicating the others), when enforce_sharding disagrees, or when partial
    # inputs are mixed. Partial inputs without linearity yield a reshard
    # suggestion that removes pending sums; with linearity all inputs must be
    # partial on the same mesh dims. Letters sharded but absent from the output
    # become pending sums. Returns an OutputSharding holding either the output
    # spec (built with tensor meta from the first input) or a reshard suggestion.
    # Raises AssertionError for shape mismatches or missing tensor_meta.
    """
    Propagate the sharding of inputs to output for ops whose data moves according to einsum notation.

    This is mostly borrowed from @zdevito's sharding simulator. Examples:
        mk,kn->mn - einsum
        ij,ij->ij - addition
        ij,j->ij - broadcasted addition
        ij->i - reduction
    Other ops could use this propagation algorithm when applied, note
    that einsum propagation only deal with list of specs (DTensor specs)
    as it only works on list of tensors!

    linearity in einop_rule means that the calling op `f` follows this rule:
        f(a + b) = f(a) + f(b)

    In this case we can propagate the partial sum, note that linearity in einop
    only applies to partial sum, not other operations like min/max (which are
    associative but not linear).
    """
    # parse einop equation and extract arg specs
    inputs, outputs = equation.split("->")
    input_dims, output_dims = inputs.split(","), outputs.split(",")
    input_specs = op_schema.args_spec
    # NOTE: only support single output unless needed in future
    output_dim = output_dims[0]

    dim_to_sharding: dict[str, int] = {}
    dim_to_size: dict[str, int] = {}
    # record pending sum, key is mesh dimension, value is pending sum
    # counter across input specs
    pending_sums_counter: dict[int, int] = {}
    seen_shardings: dict[int, str] = {}
    needs_reshard = False

    def merge_sharding(dim: str, a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Merges two mesh-dim shardings (-1 means
        # replicate) for one dim letter. Equal values are returned unchanged; a
        # replicate merges to the sharded value and sets the enclosing
        # needs_reshard flag; two different shards raise RuntimeError.
        # merge the sharding of inputs if it's able to merge, i.e. we can merge
        # replicate and shard to shard, but this will trigger an reshard operation
        if a != b:
            if a == -1 or b == -1:
                # reshard the replicate to match the sharded one
                nonlocal needs_reshard
                needs_reshard = True
                return a if a != -1 else b
            else:
                # TODO: further merge the sharding properly (i.e. reshard one input to replicate)
                raise RuntimeError(
                    f"{equation}: dim {dim} sharded two different ways: {a} and {b}"
                )
        else:
            return a

    for input_dim, input_spec in zip(input_dims, input_specs):
        # deal with partial sums
        input_sums = input_spec.sums
        for sum_dim in input_sums:
            if sum_dim not in pending_sums_counter:
                seen_shardings[sum_dim] = "+"
            # update pending sum counter for pending sum mesh
            # dimension with the occurrence from each input
            pending_sums_counter[sum_dim] = pending_sums_counter.get(sum_dim, 0) + 1

        for idx, (dim, mesh_dim) in enumerate(zip(input_dim, input_spec.dim_map)):
            if enforce_sharding and dim in enforce_sharding:
                if enforce_sharding[dim] != mesh_dim:
                    needs_reshard = True
                dim_to_sharding[dim] = enforce_sharding[dim]
                dim_to_size[dim] = input_spec.shape[idx]
            elif dim not in dim_to_sharding:
                dim_to_sharding[dim] = mesh_dim
                dim_to_size[dim] = input_spec.shape[idx]
            else:
                dim_to_sharding[dim] = merge_sharding(
                    dim, dim_to_sharding[dim], mesh_dim
                )
                if dim_to_size[dim] != input_spec.shape[idx]:
                    raise AssertionError

            # after merging sharding, we check if there're multiple
            # sharding on the same mesh dim.
            merged_sharding_for_dim = dim_to_sharding[dim]
            if merged_sharding_for_dim != -1:
                if (
                    merged_sharding_for_dim in seen_shardings
                    and dim != seen_shardings[merged_sharding_for_dim]
                ):
                    needs_reshard = True
                    seen_shardings[merged_sharding_for_dim] += dim
                else:
                    seen_shardings[merged_sharding_for_dim] = dim

    if pending_sums_counter and not linearity:
        # return reshard suggestion with no pending sum, because we already properly
        # merge the sharding, this reshard suggestion is legit to use
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, []
        )
    else:
        # It's an op that supports linearity, but not all input arguments are partial
        # we fail the sharding propagation with suggestion to make all inputs be
        # partial on the corresponding mesh dim (all inputs should be partial for
        # the mesh dims in order to execute locally and delay the sum reduction)
        for value in pending_sums_counter.values():
            if value != len(input_specs):
                needs_reshard = True

    for mesh_dim, dims in seen_shardings.items():
        if len(dims) > 1:
            # we found different input dims are being sharded on the same mesh dim
            # in order to perform local op computation, we need to reshard inputs
            # base on some simple heuristics, now we simply pick the one with least comm
            # volume. (i.e. the input with least size)
            # TODO: consider a more advanced heuristic to pick the best sharding
            costs = []
            for d in dims:
                cost = 0
                for input_dim, input_spec in zip(input_dims, input_specs):
                    if (
                        d in input_dim
                        and input_spec.dim_map[input_dim.index(d)] == mesh_dim
                    ):
                        if input_spec.tensor_meta is None:
                            raise AssertionError
                        global_shape = input_spec.tensor_meta.shape
                        local_shape, _ = compute_local_shape_and_global_offset(
                            global_shape,
                            input_spec.mesh,
                            input_spec.placements,
                            skip_offset=True,
                        )
                        cost += prod(local_shape) * input_spec.mesh.size(mesh_dim)

                costs.append(cost)
            d_to_keep_sharding = dims[costs.index(max(costs))]
            for d in dims:
                # update dim_to_sharding to keep the sharding of the dim with
                # highest comm and make the rest of the dims to replicate
                if d != d_to_keep_sharding:
                    dim_to_sharding[d] = -1

    pending_sums = list(pending_sums_counter.keys())
    if needs_reshard:
        return _gen_reshard_suggestions(
            op_schema, input_dims, input_specs, dim_to_sharding, pending_sums
        )

    # generate output pending sum if a dim is sharded, and it appears in input
    # but not output
    for dim, shard_on_mesh in dim_to_sharding.items():
        if dim not in output_dims[0] and shard_on_mesh != -1:
            pending_sums.append(shard_on_mesh)

    # if no need to reshard, we directly generate the output sharding
    output_dim_map = []
    output_shape = []
    for dim in output_dim:
        if dim == "1":
            # find output dim that is a singleton dimension, mark sharding and shape
            output_dim_map.append(-1)
            output_shape.append(1)
        else:
            output_dim_map.append(dim_to_sharding[dim])
            output_shape.append(dim_to_size[dim])

    # XXX: since we still need to have intermediate shape calculation, we need
    # to pass in the shape here. We should remove this once sharding decomp works
    # for ops like addmm
    if input_specs[0].tensor_meta is None:
        raise AssertionError
    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        input_specs[0].tensor_meta.stride,
        input_specs[0].tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_specs[0].mesh,
            output_dim_map,
            pending_sums,
            tensor_meta=tensor_meta,
        )
    )


def pointwise_rule(op_schema: OpSchema, linearity: bool = False) -> OutputSharding:
    # [FUNCTION DOC]
    # Builds an einsum equation for a pointwise op from the input specs: dim
    # letters are right-aligned to the highest rank (numpy broadcasting), and
    # size-1 or missing dims get the special letter "1" so they are ignored for
    # sharding; the output letter is "1" only if every input is singleton there.
    # For inplace ops or out-variants, the output sharding is enforced to follow
    # the first arg or the "out" kwarg. Delegates to einop_rule and returns its
    # OutputSharding; linearity is passed through.
    """
    Propagate the sharding for pointwise operations.

    Examples:
        ij,ij->ij - addition/mul
        ij,j->ij - broadcasted addition
    """
    alphabet = string.ascii_lowercase
    # find the max_dim first in case we need to broadcasting
    input_specs = op_schema.args_spec
    max_dim = max(input.ndim for input in input_specs)
    dimchars = []
    singleton_counter: list[int] = [0] * max_dim
    for input in input_specs:
        start_dim = max_dim - input.ndim
        p = alphabet[start_dim:max_dim]
        # handle the "broadcasting to a common shape case"
        # see https://pytorch.org/docs/stable/notes/broadcasting.html
        # If any of the dimensions is singleton dimension (i.e. 1).
        # we mark the dim char as a special "1" to distinguish with
        # the non-singleton dimension, so that sharding propagation
        # should just ignore the singleton dimension.
        if len(input_specs) > 1:
            for i in range(max_dim):
                if i < start_dim:
                    # treat the leading miss dim chars as singleton
                    singleton_counter[i] += 1
                elif input.shape[i - start_dim] == 1:
                    # mark singleton dim char as a special "1" in einop rule
                    singleton_counter[i] += 1
                    p = _replace_char_in_str(p, "1", (i - start_dim))

        dimchars.append(p)
    out_dimchars = alphabet[:max_dim]
    # check if we replace the all inputs dim char with singleton dimension,
    # if we replace all inputs, we also need to replace the output dimension.
    for output_dim_idx in range(len(out_dimchars)):
        if singleton_counter[output_dim_idx] == len(input_specs):
            out_dimchars = _replace_char_in_str(out_dimchars, "1", output_dim_idx)

    fmt = f"{','.join(p for p in dimchars)}->{out_dimchars}"

    enforce_sharding: dict[str, int] = {}
    if op_schema.is_inplace_op():
        follow_spec = op_schema.args_spec[0]
        enforce_sharding.update(zip(out_dimchars, follow_spec.dim_map))
    elif op_schema.is_out_variant_op():
        follow_spec = cast(DTensorSpec, op_schema.kwargs_schema["out"])
        enforce_sharding.update(zip(out_dimchars, follow_spec.dim_map))

    return einop_rule(
        fmt,
        op_schema,
        linearity=linearity,
        enforce_sharding=enforce_sharding,
    )
