# pyre-strict
# [MODULE DOC] MTIA device memory statistics API: memory_stats, max_memory_allocated,
# memory_allocated and reset_peak_memory_stats, each returning empty/zero results (or doing
# nothing) when MTIA is not yet initialized. Re-exported by the parent package. No import-time
# side effects.

r"""This package adds support for device memory management implemented in MTIA."""

from typing import Any

import torch

from . import Device, is_initialized
from ._utils import _get_device_index


def memory_stats(device: Device = None) -> dict[str, Any]:
    # [FUNCTION DOC] Returns the allocator statistics dictionary for the device (default current),
    # or an empty dictionary if MTIA is not initialized.
    r"""Return a dictionary of MTIA memory allocator statistics for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return {}
    return torch._C._mtia_memoryStats(_get_device_index(device, optional=True))


def max_memory_allocated(device: Device = None) -> int:
    # [FUNCTION DOC] Returns peak allocated bytes from the device-memory section of the stats,
    # or 0 if MTIA is not initialized.
    # KNOWN ISSUE: the lookup of the memory section defaults to the integer 0, so if the section
    # is missing the following lookup fails with AttributeError instead of returning 0.
    r"""Return the maximum memory allocated in bytes for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("peak_bytes", 0)


def memory_allocated(device: Device = None) -> int:
    # [FUNCTION DOC] Returns currently allocated bytes from the device-memory section of the
    # stats, or 0 if MTIA is not initialized.
    # KNOWN ISSUE: same default-of-integer problem as max_memory_allocated when the section is missing.
    r"""Return the current MTIA memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mtia.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("allocated_bytes", 0)


def reset_peak_memory_stats(device: Device = None) -> None:
    # [FUNCTION DOC] Resets the peak memory statistics of the device (default current); does
    # nothing if MTIA is not initialized.
    r"""Reset the peak memory stats for a given device.


    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return
    torch._C._mtia_resetPeakMemoryStats(_get_device_index(device, optional=True))


__all__ = [
    "memory_stats",
    "max_memory_allocated",
    "memory_allocated",
    "reset_peak_memory_stats",
]
