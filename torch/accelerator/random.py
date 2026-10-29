# [MODULE DOC] Accelerator-agnostic random number generator helpers: initial seed and RNG state getters for the default generator of the current accelerator. Importing has no side effects.
import torch
from torch import Tensor

from ._utils import _device_t, _get_device_index


def initial_seed(device: _device_t = None, /) -> int:
    # [FUNCTION DOC] Return the initial seed (int) of the default generator for the given device (default: current device index).
    # Resolves the index via _get_device_index (may raise ValueError/RuntimeError); eagerly initializes the accelerator runtime.
    r"""Return the initial seed of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    on the specified device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the initial seed of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        int: the initial seed of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = _get_device_index(device, optional=True)
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.initial_seed()


def get_rng_state(device: _device_t = None, /) -> Tensor:
    # [FUNCTION DOC] Return the RNG state of the default generator for the given device (default: current device) as a uint8 tensor.
    # Resolves the index via _get_device_index; eagerly initializes the accelerator runtime.
    r"""Return the RNG state of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    as a `torch.Tensor` of dtype `torch.uint8` for the specified accelerator device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the RNG state of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Tensor: the RNG state of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = _get_device_index(device, optional=True)
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    # [FUNCTION DOC] Return a list of uint8 RNG state tensors, one per device of the current accelerator (indices 0 to device_count-1).
    # Eagerly initializes the accelerator runtime.
    r"""Return a list of `torch.Tensor` of dtype `torch.uint8` representing the RNG states of all devices for
    the current :ref:`accelerator<accelerators>`.

    Returns:
        list[torch.Tensor]: the RNG states of the default generators for all devices.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    return [get_rng_state(i) for i in range(torch.accelerator.device_count())]


__all__ = ["initial_seed", "get_rng_state", "get_rng_state_all"]
