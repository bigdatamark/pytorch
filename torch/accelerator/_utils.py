# [MODULE DOC] HELPER: Internal utilities for torch.accelerator: device-index normalization and deferred (lazy) calls until the accelerator runtime is initialized. No import-time side effects.
from collections.abc import Callable

import torch
from torch.types import Device as _device_t


def _get_device_index(device: _device_t, optional: bool = False) -> int:
    # [FUNCTION DOC] HELPER: Normalize a device spec (int, string, or torch.device) to an integer device index.
    # Ints pass through. A device must match the current accelerator type, else ValueError; no accelerator raises RuntimeError.
    # If the index is missing: returns the current device index when optional is true, otherwise raises ValueError.
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    device_index: int | None = None
    if isinstance(device, torch.device):
        acc = torch.accelerator.current_accelerator()
        if acc is None:
            raise RuntimeError("Accelerator expected")
        if acc.type != device.type:
            raise ValueError(
                f"{device.type} doesn't match the current accelerator {acc}."
            )
        device_index = device.index
    if device_index is None:
        if not optional:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
        return torch.accelerator.current_device_index()
    return device_index


def _lazy_call(callable: Callable[[], None], **kwargs) -> None:
    # [FUNCTION DOC] HELPER: Run a callable once the current accelerator's runtime is initialized.
    # Delegates to the device module's _lazy_call (forwarding extra keyword arguments) if present, else runs it immediately.
    # Silently does nothing if there is no current accelerator. Returns None.
    r"""Defer a callable until the :ref:`accelerator<accelerators>` runtime is initialized.

    If the runtime is already initialized or the backend does not support lazy
    initialization, the callable runs immediately. If no accelerator is
    available, the callable is silently dropped.

    Args:
        callable (Callable[[], None]): The function to be called.
        **kwargs: Additional keyword arguments forwarded to the backend's ``_lazy_call``
            (e.g., ``seed=True``, ``seed_all=True``).
    """
    acc = torch.accelerator.current_accelerator()
    if acc is None:
        return
    device_module = torch.get_device_module(acc)
    if hasattr(device_module, "_lazy_call"):
        device_module._lazy_call(callable, **kwargs)
    else:
        # Backend does not support lazy initialization; run immediately.
        callable()
