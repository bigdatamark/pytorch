# mypy: allow-untyped-defs
# [MODULE DOC] Global random number generator controls: get/set the CPU RNG state, seed all
# devices (CPU, CUDA, MPS, XPU, MTIA and any privateuse1 custom device), query the initial
# seed, a fork_rng context manager that restores RNG state on exit, and a helper returning
# the DataLoader thread-worker generator. Import-time side effect: binds the default CPU
# generator imported from the C extension.
import contextlib
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch


__all__ = [
    "set_rng_state",
    "get_rng_state",
    "manual_seed",
    "seed",
    "initial_seed",
    "fork_rng",
    "thread_safe_generator",
]


if TYPE_CHECKING:
    from torch.utils.data._utils.worker import WorkerInfo

from torch._C import default_generator


def set_rng_state(new_state: torch.Tensor) -> None:
    # [FUNCTION DOC] Sets the state of the default CPU generator from a byte tensor
    # (as produced by get_rng_state). Only the CPU generator is affected. Returns None.
    r"""Sets the random number generator state.

    .. note:: This function only works for CPU. For CUDA, please use
        :func:`torch.manual_seed`, which works for both CPU and CUDA.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    default_generator.set_state(new_state)


def get_rng_state() -> torch.Tensor:
    # [FUNCTION DOC] Returns the default CPU generator state as a byte tensor; other
    # devices are not included.
    r"""Returns the random number generator state as a `torch.ByteTensor`.

    .. note:: The returned state is for the default generator on CPU only.

    See also: :func:`torch.random.fork_rng`.
    """
    return default_generator.get_state()


def manual_seed(seed) -> torch._C.Generator:
    # [FUNCTION DOC] Public entry point that seeds every device RNG and the default CPU
    # generator with the given integer seed (delegates to the implementation helper) and
    # returns the default generator. Raises RuntimeError for out-of-range seeds.
    r"""Sets the seed for generating random numbers on all devices. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """
    return _manual_seed_impl(seed)


def _manual_seed_impl(seed) -> torch._C.Generator:
    # [FUNCTION DOC] HELPER: Converts seed to an integer, then seeds CUDA, MPS, XPU, MTIA
    # and any custom device, each only if that backend is not in a bad forked subprocess
    # state, and finally seeds and returns the default CPU generator. Imports each backend
    # module lazily. Side effects: mutates all those global generators.
    seed = int(seed)
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return default_generator.manual_seed(seed)


def seed() -> int:
    # [FUNCTION DOC] Draws a non-deterministic seed from the default CPU generator (which
    # reseeds itself with it), applies the same seed to the CUDA, MPS, XPU, MTIA and custom
    # device generators unless in a bad fork state, and returns the 64-bit seed.
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64 bit number used to seed the RNG.
    """
    seed = default_generator.seed()
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return seed


def _seed_custom_device(seed) -> None:
    # [FUNCTION DOC] HELPER: Seeds the registered privateuse1 backend module, if torch has
    # one, by calling its manual_seed_all unless its bad-fork check reports true. If the
    # module lacks either required function, emits a UserWarning explaining the seed has no
    # effect. Does nothing if no such module exists. Returns None.
    r"""Sets the seed to generate random numbers for custom device.

    Args:
        seed (int): The desired seed.

    See [Note: support the custom device with privateuse1]
    """
    seed = int(seed)
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    if hasattr(torch, custom_backend_name):
        custom_device_mod = getattr(torch, custom_backend_name)
        _bad_fork_name = "_is_in_bad_fork"
        _seed_all_name = "manual_seed_all"
        if hasattr(custom_device_mod, _bad_fork_name) and hasattr(
            custom_device_mod, _seed_all_name
        ):
            if not getattr(custom_device_mod, _bad_fork_name)():
                getattr(custom_device_mod, _seed_all_name)(seed)
        else:
            message = f"Set seed for `{custom_backend_name}` device does not take effect, please add API's "
            message += f"`{_bad_fork_name}` and `{_seed_all_name}` to `{custom_backend_name}` device module."
            warnings.warn(message, UserWarning, stacklevel=3)


def initial_seed() -> int:
    # [FUNCTION DOC] Returns the initial seed of the default CPU generator as an integer.
    r"""Returns the initial seed for generating random numbers as a
    Python `long`.

    .. note:: The returned seed is for the default generator on CPU only.
    """
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type=None,
) -> Generator:
    # [FUNCTION DOC] Context manager that saves the CPU RNG state and that of the selected
    # devices, runs the body, and restores all saved states on exit (even on error). Does
    # nothing for a meta device type or when enabled is false. Device type defaults to the
    # current accelerator, else cuda; devices default to all of that type and a one-time
    # warning is emitted if more than one exists. Raises RuntimeError if no module is
    # registered for the device type. _caller and _devices_kw only shape the warning text.
    # KNOWN ISSUE: the warning text contains a literal placeholder for the lowercase
    # CPU-only hint because that piece is not an f-string.
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is ``None``, in which case the type
            is taken from :func:`torch.accelerator.current_accelerator`, falling back
            to ``"cuda"`` when the type cannot be determined. As for supported devices,
            see details in :ref:`accelerator<accelerators>`
    """

    if device_type == "meta":
        yield
        return

    acc = torch.accelerator.current_accelerator()
    # Default to cuda instead of CPU since CPU rng is always forked
    device_type = device_type or (acc.type if acc is not None else "cuda")

    device_mod = torch.get_device_module(device_type)
    if device_mod is None:
        raise RuntimeError(
            f"torch has no module of `{device_type}`, you should register "
            + "a module by `torch._register_device_module`."
        )
    global _fork_rng_warned_already

    # Internal arguments:
    #   _caller: the function which called fork_rng, which the user used
    #   _devices_kw: the devices keyword of _caller

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = device_mod.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            message = (
                f"{device_type.upper()} reports that you have {num_devices} available devices, and "
                f"you have used {_caller} without explicitly specifying which devices are being used. "
                f"For safety, we initialize *every* {device_type.upper()} device by default, which can "
                f"be quite slow if you have a lot of {device_type.upper()}s. If you know that you are only"
                f" making use of a few {device_type.upper()} devices, set the environment variable "
                f"{device_type.upper()}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} "
                "with the set of devices you are actually using. For example, if you are using CPU only, "
                "set device.upper()_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, "
                f"set {device_type.upper()}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices "
                f"and suppress this warning, set the '{_devices_kw}' keyword argument to "
                f"`range(torch.{device_type}.device_count())`."
            )
            warnings.warn(message, stacklevel=2)
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = torch.get_rng_state()
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            device_mod.set_rng_state(device_rng_state, device)


def thread_safe_generator() -> torch.Generator | None:
    # [FUNCTION DOC] Returns the torch generator of the current DataLoader worker when called
    # inside a thread-based worker that has an RNG; otherwise (main process, process
    # workers, no RNG) returns None so callers fall back to the global generator.
    # Imports the data utilities lazily to avoid a circular import at torch startup.
    """Returns a thread-safe random number generator for use in DataLoader workers.
    This function provides a convenient way for transforms and user code to use
    thread-safe random number generation without manually checking worker context.
    When called in a DataLoader thread worker, returns the worker's thread-local
    :class:`torch.Generator`. When called in the main process or process workers,
    returns ``None`` (which causes PyTorch functions to use the default global RNG).
    Returns:
        Optional[torch.Generator]: Thread-local generator in thread workers, None otherwise.
    Example::
        >>> from torch.random import thread_safe_generator
        >>> generator = thread_safe_generator()
        >>> torch.randint(0, 10, (5,), generator=generator)
    Example with transforms::
        >>> from torch.random import thread_safe_generator
        >>> class MyRandomTransform:
        ...     def __call__(self, img):
        ...         generator = thread_safe_generator()
        ...         offset = torch.randint(0, 10, (2,), generator=generator)
        ...         return img[..., offset[0]:, offset[1]:]
    """
    # Lazy import to avoid circular dependency during torch module initialization
    # torch.__init__ loads torch.random early, but torch.utils.data triggers
    # torch.distributed which needs torch to be fully initialized
    from torch.utils.data import get_worker_info

    worker_info: WorkerInfo | None = get_worker_info()
    if (
        worker_info is not None
        and worker_info.worker_method == "thread"
        and worker_info.rng is not None
    ):
        return worker_info.rng.torch_generator
    return None
