# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
# [MODULE DOC] Tensor-parallel convolution handlers for DTensor, for inputs sharded along the
# last (width) dimension. Provides ring halo exchange between neighboring ranks, local
# forward/backward convolution wrappers that pad or trim to account for the exchanged border
# pixels, and the two dispatch handlers (forward and backward) that run sharding propagation,
# compute locally, and wrap results as DTensor. Import-time side effect: defines the aten alias.
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.tensor._api as dtensor


aten = torch.ops.aten


def _requires_data_exchange(padding, dim_map) -> bool:
    # [FUNCTION DOC] HELPER: Decides whether neighboring ranks must exchange border data. Returns
    # false if only the batch dimension is sharded (all later dims unsharded in dim_map), else
    # true exactly when the last padding entry is nonzero.
    # Data exchange is not need if only sharded across batch dim
    if all(x == -1 for x in dim_map[1:]):
        return False
    # TODO: whether there requires data exchange is currently determined by padding
    return padding[-1] != 0


def _is_supported(input_size, kernel_size, stride, padding, dilation):
    # [FUNCTION DOC] HELPER: Validates last-dimension convolution parameters for tensor parallel
    # execution. Raises RuntimeError if dilation is not one; with padding, stride must be one
    # and half the kernel size must not exceed the input size; without padding, input size must
    # be divisible by stride and stride must equal kernel size. Returns true otherwise.
    # KNOWN ISSUE: the error message for the no-padding branch says "when there is padding".
    if dilation[-1] != 1:
        raise RuntimeError("Dilation must be 1 for tensor parallel convolution.")
    if padding[-1] != 0:
        if stride[-1] != 1:
            raise RuntimeError(
                "Stride must be 1 when there is padding for tensor parallel convolution."
            )
        if kernel_size[-1] // 2 > input_size[-1]:
            raise RuntimeError(
                "kernel_size[-1] // 2 should be less than or equal to input_size[-1] for tensor parallel convolution."
            )
    else:
        if not (input_size[-1] % stride[-1] == 0 and stride[-1] == kernel_size[-1]):
            raise RuntimeError(
                "It requires that input_size[-1] is divisible by stride[-1] and stride[-1] equals kernel_size[-1] "
                "when there is padding for tensor parallel convolution."
            )
    return True


def _ring_send_recv_construct(in_tensor, d1, d2, left, right, rank, size):
    # [FUNCTION DOC] HELPER: Halo exchange for the forward pass. Sends the last d1 columns to the
    # right neighbor and the first d2 columns to the left neighbor, receives the matching
    # border slices, waits for all requests (blocking communication), then concatenates received
    # data along the last dim: the first rank appends the right halo, the last rank prepends the
    # left halo, middle ranks do both. Returns the widened local tensor.
    # dist comms and reconstruct local input tensor
    send_to_right = in_tensor[..., -d1:].contiguous()
    send_to_left = in_tensor[..., :d2].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    if rank == 0:
        in_tensor = torch.cat([in_tensor, recv_from_right], dim=-1)
    elif rank == size - 1:
        in_tensor = torch.cat([recv_from_left, in_tensor], dim=-1)
    else:
        in_tensor = torch.cat([recv_from_left, in_tensor, recv_from_right], dim=-1)

    return in_tensor


def _ring_send_recv_aggregate(grad_in_tensor, d1, d2, left, right, rank, size):
    # [FUNCTION DOC] HELPER: Halo exchange for the input gradient in backward. Sends edge column
    # slices of the gradient to neighbors, receives theirs, waits, then trims the halo columns
    # off the gradient (by rank position) and adds the received slices into the remaining edge
    # columns, mutating the sliced view in place.
    # KNOWN ISSUE: the function has no return statement, so it returns None and the caller
    # replaces the input gradient with None; the trimmed tensor is lost to the caller.
    # dist comms and aggregate gradients for edge pixels
    send_to_right = grad_in_tensor[:, :, :, -d2:].contiguous()
    send_to_left = grad_in_tensor[:, :, :, :d1].contiguous()
    recv_from_right = torch.zeros_like(send_to_left)
    recv_from_left = torch.zeros_like(send_to_right)

    send_op_right = dist.P2POp(dist.isend, send_to_right, right)
    send_op_left = dist.P2POp(dist.isend, send_to_left, left)
    recv_op_right = dist.P2POp(dist.irecv, recv_from_right, right)
    recv_op_left = dist.P2POp(dist.irecv, recv_from_left, left)

    reqs = dist.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_left, recv_op_right]
    )
    for req in reqs:
        req.wait()

    if rank == 0:
        grad_in_tensor = grad_in_tensor[:, :, :, :-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
    elif rank == size - 1:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:]
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )
    else:
        grad_in_tensor = grad_in_tensor[:, :, :, d1:-d2]
        grad_in_tensor[:, :, :, -d1:] = torch.add(
            grad_in_tensor[:, :, :, -d1:], recv_from_right
        )
        grad_in_tensor[:, :, :, :d2] = torch.add(
            grad_in_tensor[:, :, :, :d2], recv_from_left
        )


def tp_convolution(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
    dim_map: list[int],
) -> object:
    # [FUNCTION DOC] Runs the local part of a width-sharded convolution. Requires the op to be the
    # default aten convolution with nine arguments and a list padding, else raises
    # AssertionError. If no data exchange is needed, calls the op directly. Otherwise validates
    # parameters (AssertionError note: _is_supported raises RuntimeError on unsupported ones),
    # exchanges halos with ring neighbors, runs the op on the widened input, and trims padding
    # columns from the output by rank position. Returns the local result. Uses collective
    # communication (side effect).
    if op_call != aten.convolution.default:
        raise AssertionError
    if len(local_tensor_args) != 9:
        raise AssertionError

    rank = dist.get_rank()
    size = dist.get_world_size()
    in_tensor = cast(torch.Tensor, local_tensor_args[0])
    weight = cast(torch.Tensor, local_tensor_args[1])
    stride, padding, dilation = local_tensor_args[3:6]

    if not isinstance(padding, list):
        raise AssertionError

    if not _requires_data_exchange(padding, dim_map):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        if not _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation):
            raise AssertionError(
                "tp_convolution data exchange requires supported stride/padding/dilation"
            )
        # step 0 compute the overlap pixels of the input tensor
        d = weight.shape[-1] - 1
        d1 = d // 2
        d2 = d - d1
        if d1 + d2 != d:
            raise AssertionError
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 feed local input tensor to op_call
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = in_tensor
        local_tensor_args = cast(tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step3 remove extra outputs from the results
        padding_w = padding[-1]
        w = local_results.size(-1)
        if rank == 0:
            local_results = local_results[..., : w - padding_w]
        elif rank == size - 1:
            local_results = local_results[..., padding_w:]
        else:
            local_results = local_results[..., padding_w : w - padding_w]

        return local_results


def tp_convolution_backward(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
    dim_map: list[int],
) -> object:
    # [FUNCTION DOC] Runs the local part of the backward of a width-sharded convolution. Requires
    # the default aten convolution_backward with eleven arguments and a list padding, else
    # AssertionError. Without data exchange, calls the op directly. Otherwise exchanges input
    # halos, pads the output gradient by rank position, runs the op, and aggregates the input
    # gradient edges across neighbors. Returns the tuple of local gradient results.
    # KNOWN ISSUE: the aggregate helper returns None, so the input gradient entry becomes None;
    # also weight width is read from a fixed dim index and padding from a fixed index, assuming 4D.
    if op_call != aten.convolution_backward.default:
        raise AssertionError
    if len(local_tensor_args) != 11:
        raise AssertionError

    rank = dist.get_rank()
    size = dist.get_world_size()
    grad_out_tensor = cast(torch.Tensor, local_tensor_args[0])
    in_tensor = cast(torch.Tensor, local_tensor_args[1])
    weight = cast(torch.Tensor, local_tensor_args[2])
    stride, padding, dilation = local_tensor_args[4:7]

    if not isinstance(padding, list):
        raise AssertionError

    if not _requires_data_exchange(padding, dim_map):
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)
        return local_results
    else:
        if not _is_supported(in_tensor.shape, weight.shape, stride, padding, dilation):
            raise AssertionError(
                "tp_convolution_backward data exchange requires supported stride/padding/dilation"
            )
        # step 0 compute the overlap pixels of the input tensor
        d = weight.shape[3] - 1
        d1 = d // 2
        d2 = d - d1
        if d1 + d2 != d:
            raise AssertionError
        right = (rank + 1) % size
        left = (rank - 1 + size) % size

        # step1 reconstruct local input tensor
        in_tensor = _ring_send_recv_construct(
            in_tensor, d1, d2, left, right, rank, size
        )

        # step2 reconstruct local gradient output tensor
        padding_w = padding[1]
        if rank == 0:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (0, padding_w), "constant", 0
            )
        elif rank == size - 1:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, 0), "constant", 0
            )
        else:
            grad_out_tensor = torch.nn.functional.pad(
                grad_out_tensor, (padding_w, padding_w), "constant", 0
            )

        # step3 feed local input tensor to op_call
        local_tensor_args_list = list(local_tensor_args)
        local_tensor_args_list[0] = grad_out_tensor
        local_tensor_args_list[1] = in_tensor
        local_tensor_args = cast(tuple[object, ...], local_tensor_args_list)
        local_results = op_call(*local_tensor_args, **local_tensor_kwargs)

        # step4 aggregate gradients for edge pixels
        grad_in_tensor = local_results[0]
        if grad_in_tensor is not None:
            grad_in_tensor = _ring_send_recv_aggregate(
                grad_in_tensor, d1, d2, left, right, rank, size
            )
            local_results = list(local_results)
            local_results[0] = grad_in_tensor

        local_results = cast(tuple[object, ...], local_results)

        return local_results


def convolution_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # [FUNCTION DOC] Dispatch handler for the aten convolution on DTensors. Unwraps arguments into
    # an op info, runs sharding propagation, requires a resolved output spec (AssertionError
    # otherwise), computes the local result with the tensor parallel convolution using the
    # output spec's dim map, and wraps the result as a DTensor with that spec.
    # extract local tensor and sharding infos to an OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    output_spec = output_sharding.output_spec
    if not isinstance(output_spec, dtensor.DTensorSpec):
        raise AssertionError

    # local propagation
    local_results = tp_convolution(
        op_call,
        tuple(op_info.local_args),
        op_info.local_kwargs,
        output_spec.dim_map,
    )

    return dtensor.DTensor._op_dispatcher.wrap(local_results, output_spec)


def convolution_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    # [FUNCTION DOC] Dispatch handler for aten convolution_backward on DTensors. Requires the first
    # two arguments (grad output, input) to be DTensors (AssertionError otherwise), redistributes
    # the grad output to the input's mesh and placements (communication), runs sharding
    # propagation, computes local results with the tensor parallel backward using the input
    # spec's dim map, and wraps them with the propagated output spec.
    # Redistribute grad_output tensor to the same placement as input tensor
    # pyrefly: ignore [bad-assignment]
    args = list(args)
    if not (
        isinstance(args[0], dtensor.DTensor) and isinstance(args[1], dtensor.DTensor)
    ):
        raise AssertionError
    # pyrefly: ignore [unsupported-operation]
    args[0] = args[0].redistribute(args[1].device_mesh, args[1].placements)
    args = tuple(args)

    # extract local tensor and sharding infos to an OpInfo
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise AssertionError("output sharding should not be None")
    if not isinstance(op_info.flat_args_schema[0], dtensor.DTensorSpec):
        raise AssertionError

    # local propagation
    local_results = tp_convolution_backward(
        op_call,
        tuple(op_info.local_args),
        op_info.local_kwargs,
        op_info.flat_args_schema[0].dim_map,
    )

    return dtensor.DTensor._op_dispatcher.wrap(
        local_results, output_sharding.output_spec
    )
