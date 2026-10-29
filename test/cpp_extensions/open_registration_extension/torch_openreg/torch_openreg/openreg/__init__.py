# [PACKAGE DOC] torch_openreg.openreg: the Python device module for the "openreg" backend,
# registered with torch as torch.openreg. Provides device selection (device context manager,
# set_device, current_device, device_count), availability and lazy initialization helpers,
# and the RNG API re-exported from the random submodule. Importing it also imports the meta
# kernels (registering them) and the AMP dtype helper. Fork-safety is tracked through the
# native _isInBadFork hook.
import torch

import torch_openreg._C  # type: ignore[misc]
from . import meta
from .amp import get_amp_supported_dtype


_initialized = False
_is_in_bad_fork = getattr(torch_openreg._C, "_isInBadFork", lambda: False)


class device:
    # [CLASS DOC] Context manager that temporarily selects an openreg device. State: idx, the
    # requested device index (negative or None-derived means no change) and prev_idx, the
    # device that was current before entering (initially -1). Key methods: __enter__ swaps in
    # the requested device, __exit__ restores the previous one and never suppresses errors.
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        # [FUNCTION DOC] Resolves the given device (torch.device, int or None) to an index via
        # the accelerator helper, allowing an unspecified device, and records it in idx.
        # Initializes prev_idx to -1. No device change happens until __enter__.
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        # [FUNCTION DOC] Makes the requested device current via the native exchange call and
        # stores the previously current index in prev_idx. Returns nothing (so "as" targets
        # are None). Side effect: changes the native current device.
        self.prev_idx = torch_openreg._C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        # [FUNCTION DOC] Restores the device that was current on entry and returns False so
        # exceptions propagate. Side effect: changes the native current device.
        # KNOWN ISSUE: overwrites self.idx with the native set-device return value (likely
        # None), which looks unintended and makes re-entering the same object unreliable.
        self.idx = torch_openreg._C._set_device(self.prev_idx)
        return False


def is_available():
    # [FUNCTION DOC] Always returns True; the openreg backend is considered present.
    return True


def device_count() -> int:
    # [FUNCTION DOC] Returns the number of openreg devices reported by the native module.
    return torch_openreg._C._get_device_count()


def current_device():
    # [FUNCTION DOC] Returns the index of the current openreg device from the native module.
    return torch_openreg._C._get_device()


# LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
def set_device(device) -> None:
    # [FUNCTION DOC] Selects the given device index as current; negative values are ignored.
    # Side effect: changes the native current device. Expects an integer index.
    if device >= 0:
        torch_openreg._C._set_device(device)


# LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION


def init():
    # [FUNCTION DOC] Public entry point that forces initialization of the openreg runtime
    # (delegates to the lazy initializer). Raises an error in a forked subprocess.
    _lazy_init()


def is_initialized():
    # [FUNCTION DOC] Returns True if the runtime was initialized in this process and the
    # process is not in a bad fork state.
    return _initialized and not _is_in_bad_fork()


def _lazy_init():
    # [FUNCTION DOC] HELPER: Initializes the native runtime once. Does nothing if already
    # initialized; raises a runtime error if called in a bad forked subprocess (advising the
    # spawn start method). Side effect: sets the module-level initialized flag.
    global _initialized
    if is_initialized():
        return
    if _is_in_bad_fork():
        raise RuntimeError(
            "Cannot re-initialize OpenReg in forked subprocess. To use OpenReg with "
            "multiprocessing, you must use the 'spawn' start method"
        )
    torch_openreg._C._init()
    _initialized = True


from .random import *  # noqa: F403


__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
]
