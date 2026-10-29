# mypy: allow-untyped-defs
# [MODULE DOC] XPU random number generator control: get/set RNG state and seeding for the
# current device or all devices, using the per-device default generators in torch.xpu.
# Seeding and state setting are deferred through the lazy call queue when XPU is not yet
# initialized. Exports the names listed in __all__.
from collections.abc import Iterable

import torch
from torch import Tensor

from . import _lazy_call, _lazy_init, current_device, device_count, is_initialized


def get_rng_state(device: int | str | torch.device = "xpu") -> Tensor:
    # [FUNCTION DOC] Returns the default generator state (byte tensor) of device (int, string
    # or torch.device; no index means current device). Eagerly initializes XPU.
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    # [FUNCTION DOC] Returns a list with the RNG state of each device, indexed by device number.
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: int | str | torch.device = "xpu") -> None:
    # [FUNCTION DOC] Sets the default generator state of device to new_state. If XPU is not
    # yet initialized, the state is first cloned (outside functorch). The actual set is
    # scheduled through the lazy call mechanism, resolving a missing index at run time.
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    if not is_initialized():
        with torch._C._DisableFuncTorch():
            new_state = new_state.clone(memory_format=torch.contiguous_format)

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)

    def cb() -> None:
        # [FUNCTION DOC] HELPER: Deferred action applying new_state to the generator of the
        # resolved device index (current device if none).
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    # [FUNCTION DOC] Sets the RNG state of device i to the i-th item of new_states, in order.
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    # [FUNCTION DOC] Seeds the current device's generator with the integer seed. Deferred
    # via the lazy call queue (flagged as a seed call), so it is a no-op without XPU.
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb() -> None:
        # [FUNCTION DOC] HELPER: Deferred action seeding the current device's generator.
        idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    # [FUNCTION DOC] Seeds every device's generator with the integer seed, deferred via the
    # lazy call queue (flagged as seed-all), so it is a no-op without XPU.
    r"""Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb() -> None:
        # [FUNCTION DOC] HELPER: Deferred action seeding all device generators with seed.
        for i in range(device_count()):
            default_generator = torch.xpu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    # [FUNCTION DOC] Seeds the current device's generator with a nondeterministic random seed,
    # deferred via the lazy call queue. Returns nothing.
    r"""Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb() -> None:
        # [FUNCTION DOC] HELPER: Deferred action randomly seeding the current device's generator.
        idx = current_device()
        default_generator = torch.xpu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    # [FUNCTION DOC] Seeds all devices with one random seed: the first device is randomly
    # seeded and the others are manually seeded with its resulting initial seed. Deferred
    # via the lazy call queue.
    r"""Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if XPU is not available; in that case, it is silently ignored.
    """

    def cb() -> None:
        # [FUNCTION DOC] HELPER: Deferred action seeding the first device randomly and copying
        # its initial seed to the remaining devices.
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch.xpu.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed() -> int:
    # [FUNCTION DOC] Returns the initial seed of the current device's generator. Eagerly
    # initializes XPU.
    r"""Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch.xpu.default_generators[idx]
    return default_generator.initial_seed()


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]
