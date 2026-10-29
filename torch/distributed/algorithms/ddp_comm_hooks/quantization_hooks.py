# mypy: allow-untyped-defs
# [MODULE DOC] Experimental DDP communication hooks that quantize gradients to uint8 (per tensor or per channel)
# and exchange them with allgather, then dequantize and average locally. Includes the quantize/dequantize
# backends and an allgather output-list helper. No import-time side effects.
import torch
import torch.distributed as dist
from torch import nn


def _quantize_per_tensor_backend(x, scale, zero_point):
    # [FUNCTION DOC] HELPER: Quantize x with a scalar scale and zero point: divide, round, shift, clamp to 0..255, cast to uint8.
    y = torch.round(x / scale) + zero_point
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_tensor_backend(y, scale, zero_point):
    # [FUNCTION DOC] HELPER: Inverse of per-tensor quantization: cast to float32, subtract zero point, multiply by scale.
    x = scale * (y.to(torch.float32) - zero_point)
    return x


def _quantize_per_channel_backend(x, scale, zero_point):
    # [FUNCTION DOC] HELPER: Quantize a 2D tensor row by row with per-row scale and zero point; clamp to 0..255 and return uint8.
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, 0, 255).to(torch.uint8)
    return y


def _dequantize_per_channel_backend(y, scale, zero_point):
    # [FUNCTION DOC] HELPER: Dequantize a 2D uint8 tensor row by row using per-row scale and zero point; returns float32.
    y = y.to(torch.float32).to(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


def _get_allgather_out_list(all_gather_in_list, world_size):
    # [FUNCTION DOC] HELPER: Return a list of world_size zero tensors matching the shape, device and dtype of the input, used as allgather output buffers.
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list


def quantization_pertensor_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] Experimental DDP hook: computes min/max-observer scale and zero point of the bucket buffer,
    # allgathers them, quantizes the local buffer to uint8, allgathers the quantized tensors, then dequantizes
    # each rank's tensor with that rank's parameters and returns their mean as a future. Group defaults to WORLD.
    # Works only on flattened buffers; slower than allreduce.
    """
    Apply ``torch.quantize_per_tensor`` logic to DDP using ``allgather`` protocol.

    Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    myObserver = torch.ao.quantization.MinMaxObserver().to(tensor.device)
    myObserver(tensor)

    s, z = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)

    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # [FUNCTION DOC] HELPER: After scale/zero point exchange, quantize the local buffer with this rank's qparams and allgather it; blocks and returns the gathered result.
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their own ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_tensor_backend(
            tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1]
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        # [FUNCTION DOC] HELPER: Dequantize each rank's gathered tensor with that rank's qparams, sum in float32 and divide by world size.
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return aggregated_dequantized_tensor / world_size

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)


def quantization_perchannel_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket, bucket_size=512
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] Experimental DDP hook like the per-tensor hook but quantizing per channel: the buffer is
    # zero-padded and reshaped into rows of bucket_size elements, a per-channel observer computes per-row scale and zero
    # point, which are allgathered along with the quantized rows. Result is dequantized, summed, flattened, trimmed
    # to the original length and divided by world size. Returns a future.
    # KNOWN ISSUE: padding adds a full extra row of zeros when the length is already a multiple of bucket_size.
    """
    Apply``torch.quantize_per_channel`` logic to DDP using ``allgather`` protocol.

    Compared to per-tensor, the main motivation of per-channel is
    for considerably large tensors such as a tensor that contains 6 million
    elements quantizing per a bucket size of 512 (or 128) elements may significantly
    increase the resolution.

    It first splits ``GradBucket`` tensor into multiple chunks (channels) of ``bucket_size``
    elements. Then, workers allgather the scales and zero points of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_perchannel_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    tensor_in_channels = (
        nn.functional.pad(
            input=tensor,
            pad=(0, bucket_size - len(tensor) % bucket_size),
            mode="constant",
            value=0,
        )
        .view(-1, bucket_size)
        .to(tensor.device)
    )

    myPerChannelObserver = torch.ao.quantization.PerChannelMinMaxObserver().to(
        tensor.device
    )
    myPerChannelObserver(tensor_in_channels)

    s_ch, z_ch = myPerChannelObserver.calculate_qparams()
    s_and_z = torch.stack((s_ch, z_ch)).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # [FUNCTION DOC] HELPER: Quantize the local rows with this rank's per-channel qparams and allgather them; blocks and returns the gathered result.
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their corresponding ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_channel_backend(
            tensor_in_channels,
            all_ranks_s_and_z[rank, 0, :],
            all_ranks_s_and_z[rank, 1, :],
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        # [FUNCTION DOC] HELPER: Dequantize every rank's rows with its qparams, sum, flatten, trim to original length and divide by world size.
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return (
            torch.flatten(aggregated_dequantized_tensor).to(tensor.device)[
                : tensor.size()[0]
            ]
            / world_size
        )

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)
