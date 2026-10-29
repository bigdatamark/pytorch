# [MODULE DOC] RNG API for the openreg device: get/set the default generator state,
# query the initial seed, and seed the current or all devices. Each function resolves the
# per-device default generator from the native module. Names are re-exported by the package.
import torch

import torch_openreg._C  # type: ignore[misc]
from . import _lazy_init, current_device, device_count


__all__ = [
    "get_rng_state",
    "set_rng_state",
    "manual_seed",
    "manual_seed_all",
    "initial_seed",
]


def get_rng_state(device="openreg"):
    # [FUNCTION DOC] Returns the default generator state of a device. The device may be a
    # string, an int index or a torch.device; a missing index means the current device.
    # Returns the opaque state object from the native generator.
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("openreg", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.get_state()


def set_rng_state(new_state, device="openreg"):
    # [FUNCTION DOC] Sets the default generator state of a device (resolved as in
    # get_rng_state, defaulting to the current device) to new_state. Mutates the generator.
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("openreg", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    default_generator.set_state(new_state)


def initial_seed() -> int:
    # [FUNCTION DOC] Initializes the runtime lazily, then returns the initial seed of the
    # current device's default generator.
    _lazy_init()
    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.initial_seed()


# LITERALINCLUDE START: OPENREG MANUAL SEED
def manual_seed(seed: int) -> None:
    # [FUNCTION DOC] Seeds the current device's default generator with the integer seed
    # (converted to int). Mutates generator state; does not initialize the runtime lazily.
    seed = int(seed)

    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    default_generator.manual_seed(seed)


# LITERALINCLUDE END: OPENREG MANUAL SEED


def manual_seed_all(seed: int) -> None:
    # [FUNCTION DOC] Seeds the default generator of every openreg device with the integer
    # seed (converted to int), looping over all device indices. Mutates generator state.
    seed = int(seed)

    for idx in range(device_count()):
        default_generator = torch_openreg._C._get_default_generator(idx)
        default_generator.manual_seed(seed)
