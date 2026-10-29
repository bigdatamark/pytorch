# [MODULE DOC] Helpers for the foreach/fused optimizer kernels: lists of device types that
# support them, a routine that groups tensors by device and dtype, and support predicates.
# Defines the aliases TensorListList and Indices and the list of supported tensor types.
from typing import TypeAlias

import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad


def _get_foreach_kernels_supported_devices() -> list[str]:
    # [FUNCTION DOC] HELPER: Returns a new list of device type names with foreach kernels
    # (cuda, xpu, mtia and the registered privateuse1 backend name). Cpu is not included.
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", "mtia", torch._C._get_privateuse1_backend_name()]


def _get_fused_kernels_supported_devices() -> list[str]:
    # [FUNCTION DOC] HELPER: Returns a new list of device type names with fused optimizer
    # kernels (mps, cuda, xpu, hpu, cpu, mtia and the privateuse1 backend name).
    r"""Return the device type list that supports fused kernels in optimizer."""
    return [
        "mps",
        "cuda",
        "xpu",
        "hpu",
        "cpu",
        "mtia",
        torch._C._get_privateuse1_backend_name(),
    ]


TensorListList: TypeAlias = list[list[Tensor | None]]
Indices: TypeAlias = list[int]
_foreach_supported_types = [torch.Tensor]


# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
# with_indices (bool, optional): whether to track previous indices as the last list per dictionary entry.
#   It comes in handy if there are Nones or literals in the tensorlists that are getting scattered out.
#   Whereas mutating a tensor in the resulting split-up tensorlists WILL propagate changes back to the
#   original input tensorlists, changing up Nones/literals WILL NOT propagate, and manual propagation
#   may be necessary. Check out torch/optim/sgd.py for an example.
@no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]:
    # [FUNCTION DOC] HELPER: Under no-grad, delegates to the C++ implementation to split a list
    # of tensor lists into groups keyed by (device, dtype). Each value is the regrouped tensor
    # lists plus the original indices (meaningful when with_indices is true). Tensors in the
    # groups alias the inputs. Input assumptions are not verified.
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)


def _device_has_foreach_support(device: torch.device) -> bool:
    # [FUNCTION DOC] HELPER: True when the device type is cpu or in the foreach-capable list and
    # the code is not being scripted with TorchScript; otherwise False.
    return (
        device.type in (_get_foreach_kernels_supported_devices() + ["cpu"])
        and not torch.jit.is_scripting()
    )


def _has_foreach_support(tensors: list[Tensor], device: torch.device) -> bool:
    # [FUNCTION DOC] HELPER: True when the device supports foreach and every entry is either
    # None or exactly a plain Tensor (subclasses are rejected because the type check is exact).
    return _device_has_foreach_support(device) and all(
        t is None or type(t) in _foreach_supported_types for t in tensors
    )
