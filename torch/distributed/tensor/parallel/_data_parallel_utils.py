# [MODULE DOC] HELPER: Utilities used when composing Tensor Parallel with data parallelism
# (DDP/FSDP): a gradient hook that waits on async collective tensors, and functions that
# flatten a DTensor into its local tensor plus spec and rebuild a DTensor from them.
# No import-time side effects.
from functools import partial
from typing import no_type_check

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec


@no_type_check
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None):
    # [FUNCTION DOC] HELPER: Gradient hook. If grad is an AsyncCollectiveTensor, blocks on its
    # result (via wait), running the wait inside the given compute stream context of
    # device_handle when compute_stream is not None. Otherwise returns grad unchanged.
    # Returns the resolved gradient. No other side effects.
    if isinstance(grad, AsyncCollectiveTensor):
        if compute_stream is not None:
            with device_handle.stream(compute_stream):
                grad = grad.wait()
        else:
            grad = grad.wait()

    return grad


def _flatten_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, DTensorSpec | None]:
    # [FUNCTION DOC] HELPER: For a DTensor, marks its local tensor as requiring grad (in place
    # mutation) and returns the pair (local tensor, DTensor spec). For a plain tensor returns
    # the tensor and None.
    if isinstance(tensor, DTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


@no_type_check
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None):
    # [FUNCTION DOC] HELPER: Rebuilds a DTensor from a local tensor and a saved spec (mesh,
    # placements, global shape and stride) without a consistency check. If the local tensor
    # requires grad, registers sync_grad_hook on it (bound to device_handle and compute_stream)
    # so async gradient collectives are waited on. Returns the DTensor.
    # unflatten would mainly be called every time FSDP allgather parameters.
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
        shape=spec.shape,
        stride=spec.stride,
    )
    if tensor.requires_grad:
        # only register the hook if the tensor requires grad
        tensor.register_hook(
            partial(
                sync_grad_hook,
                device_handle=device_handle,
                compute_stream=compute_stream,
            )
        )
    return result
