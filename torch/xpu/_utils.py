# [MODULE DOC] HELPER: XPU-specific wrapper around the shared device index resolver.
# Exposes _get_device_index, which validates that a device is XPU (or CPU if allowed).
from typing import Any

import torch

# The _get_device_index has been moved to torch.utils._get_device_index
from torch._utils import _get_device_index as _torch_get_device_index


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    # [FUNCTION DOC] HELPER: Resolves device (int, string, torch.device, xpu device object,
    # or None) to an integer XPU index. Ints are returned unchanged; strings are parsed.
    # Raises ValueError if a torch.device is not xpu (or cpu when allow_cpu is set).
    # Otherwise delegates to the generic resolver, which handles None/optional/allow_cpu.
    r"""Get the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a XPU device. Note that for a XPU device without a specified index,
    i.e., ``torch.device('xpu')``, this will return the current default XPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default XPU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["xpu", "cpu"]:
                raise ValueError(f"Expected a xpu or cpu device, but got: {device}")
        elif device.type != "xpu":
            raise ValueError(f"Expected a xpu device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.xpu.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)
