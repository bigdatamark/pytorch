# mypy: allow-untyped-defs
# [MODULE DOC] Generic DDP communication hooks: allreduce averaging, fp16/bf16 compression hooks, and
# wrappers that add fp16/bf16 compression around any other hook. Exports the five public names in __all__.
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist


__all__ = [
    "allreduce_hook",
    "fp16_compress_hook",
    "bf16_compress_hook",
    "fp16_compress_wrapper",
    "bf16_compress_wrapper",
]


def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Divide tensor in place by the group size (WORLD if group is None), then launch an
    # async all_reduce and return a future resolving to the reduced tensor. Mutates tensor.
    """Average the input gradient tensor by allreduce and returns a future."""
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    # pyrefly: ignore [missing-attribute]
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] DDP hook that averages the bucket buffer across the process group via allreduce and returns
    # a future of the result; equivalent to default DDP behavior.
    """
    Call ``allreduce`` using ``GradBucket`` tensors.

    Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result.

    If user registers this DDP communication hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    return _allreduce_fut(process_group, bucket.buffer())


def _compress_hook(
    dtype: torch.dtype,
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Cast the bucket buffer (or first element if given a tuple) to dtype, divide by world
    # size, allreduce it (functional collective when compiling, else async), then copy the result back in place into
    # the original buffer. Returns a future (or tensor when compiling) of the decompressed buffer.
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    buffer = (
        cast(tuple[torch.Tensor, ...], bucket)[0]
        if isinstance(bucket, tuple)
        else bucket.buffer()
    )
    compressed_tensor = buffer.to(dtype).div_(world_size)

    def decompress(fut):
        # [FUNCTION DOC] HELPER: Copy the reduced value (tensor or future result) into the original buffer in place and return it.
        decompressed_tensor = buffer
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        value = fut if isinstance(fut, torch.Tensor) else fut.value()[0]
        decompressed_tensor.copy_(value)
        return decompressed_tensor

    if torch.compiler.is_compiling():
        grad = dist._functional_collectives.all_reduce(
            compressed_tensor,
            "sum",
            # pyrefly: ignore [bad-argument-type]
            group_to_use,
        )
        return decompress(grad)
    else:
        fut = dist.all_reduce(
            compressed_tensor, group=group_to_use, async_op=True
        ).get_future()
        return fut.then(decompress)


def fp16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] Compression hook: allreduce the bucket in float16 averaged by group size, then cast back in place.
    """
    Compress by casting ``GradBucket`` to ``torch.float16`` divided by process group size.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    return _compress_hook(torch.float16, process_group, bucket)


def bf16_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    # [FUNCTION DOC] Compression hook like the fp16 one but using bfloat16 (experimental; needs NCCL > 2.9.6).
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (``torch.bfloat16``)
    and then divides it by the process group size.
    It allreduces those ``bfloat16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    return _compress_hook(torch.bfloat16, process_group, bucket)


def fp16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    # [FUNCTION DOC] Return a hook that casts the bucket buffer to float16, runs the wrapped hook, and copies the
    # result back in place to the original dtype. Takes a hook callable (state, bucket) returning a future.
    """
    Cast input tensor to ``torch.float16``, cast result of hook back to input dtype.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    floating point format (``torch.float16``), and casts the resulting tensor of the given hook back to
    the input data type, such as ``float32``.
    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
    """

    def fp16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Replace the bucket buffer with a float16 copy (mutates bucket), run the wrapped
        # hook, and chain a decompress step. Returns the resulting future.
        # Cast bucket tensor to FP16.
        bucket.set_buffer(bucket.buffer().to(torch.float16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            # [FUNCTION DOC] HELPER: Copy the wrapped hook's result into the bucket buffer in place and return it.
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return fp16_compress_wrapper_hook


def bf16_compress_wrapper(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    # [FUNCTION DOC] Like fp16_compress_wrapper but casts to bfloat16 (experimental; needs NCCL > 2.9.6).
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_  (``torch.bfloat16``),
    and casts the resulting tensor of the given hook back to the input data type, such as ``float32``.

    Therefore, ``bf16_compress_hook`` is equivalent to ``bf16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
    """

    def bf16_compress_wrapper_hook(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Replace the bucket buffer with a bfloat16 copy (mutates bucket), run the wrapped
        # hook, and chain a decompress step. Returns the resulting future.
        # Cast bucket tensor to BF16.
        bucket.set_buffer(bucket.buffer().to(torch.bfloat16))

        fut = hook(hook_state, bucket)

        def decompress(fut):
            # [FUNCTION DOC] HELPER: Copy the wrapped hook's result into the bucket buffer in place and return it.
            decompressed_tensor = bucket.buffer()
            # Decompress in place to reduce the peak memory.
            # See: https://github.com/pytorch/pytorch/issues/45968
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor

        # Decompress after hook has run.
        return fut.then(decompress)

    return bf16_compress_wrapper_hook
