# mypy: allow-untyped-defs
# [PACKAGE DOC] torch.mps: Python interface to the MPS (Metal) backend on Apple GPUs. Provides
# device count and availability checks, synchronization, RNG state/seed control through a lazily
# fetched default generator, caching-allocator and memory statistics, a memory fraction limit,
# runtime shader compilation and metallib loading, and a private host-alias storage helper.
# Re-exports the Event class and the profiler submodule. Exports are listed in __all__.
# Import-time side effect: reads the optional bad-fork hook from the C extension; no device work.
r"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""

import torch
from torch import Tensor


_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]


# local helper function (not public or exported)
def _get_default_mps_generator() -> torch._C.Generator:
    # [FUNCTION DOC] HELPER: Returns the process-wide default MPS generator, fetching it from the
    # C extension on first use and caching it in a module global.
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator


def device_count() -> int:
    # [FUNCTION DOC] Returns 1 if PyTorch was built with MPS and an MPS device is available, else
    # 0 (at most one MPS device is reported).
    r"""Returns the number of available MPS devices."""
    return int(torch._C._has_mps and torch._C._mps_is_available())


def synchronize() -> None:
    # [FUNCTION DOC] Blocks until all work queued on the MPS device has completed.
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_deviceSynchronize()


def get_rng_state(device: int | str | torch.device = "mps") -> Tensor:
    # [FUNCTION DOC] Returns the state of the default MPS generator as a byte tensor.
    # KNOWN ISSUE: the device argument is accepted but ignored; there is only one generator.
    r"""Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    return _get_default_mps_generator().get_state()


def set_rng_state(new_state: Tensor, device: int | str | torch.device = "mps") -> None:
    # [FUNCTION DOC] Restores the default MPS generator from a byte tensor state (a contiguous
    # copy is made first, so the input is not modified). Mutates generator state.
    # KNOWN ISSUE: the device argument is accepted but ignored.
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)


def manual_seed(seed: int) -> None:
    # [FUNCTION DOC] Seeds the default MPS generator with the integer seed (converted with int).
    # Silently does nothing when PyTorch was built without MPS, because the global manual seed
    # function calls it unconditionally. Mutates generator state.
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # the torch.mps.manual_seed() can be called from the global
    # torch.manual_seed() in torch/random.py. So we need to make
    # sure mps is available (otherwise we just return without
    # erroring out)
    if not torch._C._has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)


def seed() -> None:
    # [FUNCTION DOC] Seeds the default MPS generator with a non-deterministic random seed.
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()


def empty_cache() -> None:
    # [FUNCTION DOC] Releases cached, unused memory held by the MPS allocator back to the system.
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()


def set_per_process_memory_fraction(fraction) -> None:
    # [FUNCTION DOC] Limits this process's MPS allocations to a fraction (0 to 2 inclusive) of the
    # recommended maximum device memory; 0 means unlimited. Raises TypeError if fraction is not a
    # float and ValueError if outside the range. Side effect: configures the allocator.
    # KNOWN ISSUE: an int such as 0 is rejected with TypeError even though the docstring says
    # passing 0 means unlimited; only a float works.
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 2:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~2")

    torch._C._mps_setMemoryFraction(fraction)


def current_allocated_memory() -> int:
    # [FUNCTION DOC] Returns bytes currently occupied by tensors (excluding allocator caches).
    r"""Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()


def driver_allocated_memory() -> int:
    # [FUNCTION DOC] Returns bytes allocated by the Metal driver for this process, including
    # allocator caches and MPS framework allocations.
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()


def recommended_max_memory() -> int:
    # [FUNCTION DOC] Returns the recommended maximum GPU working set size in bytes reported by Metal.
    r"""Returns recommended max Working set size for GPU memory in bytes.

    .. note::
       Recommended max working set size for Metal.
       returned from device.recommendedMaxWorkingSetSize.
    """
    return torch._C._mps_recommendedMaxMemory()


def compile_shader(source: str):
    # [FUNCTION DOC] Compiles Metal shader source text and returns a library object whose kernels
    # can be called from Python. Raises RuntimeError if the build lacks MPS shader compilation.
    # Header includes in the source are expanded first, searching the include directory of the
    # package. Side effect: compiles on the device.
    r"""Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
        >>> lib = torch.mps.compile_shader(
        ... "kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }"
        ...  )
        >>> x = torch.zeros(16, device="mps")
        >>> lib.full(x, 3.14)
    """
    from pathlib import Path

    from torch.utils._cpp_embed_headers import _embed_headers

    if not hasattr(torch._C, "_mps_compileShader"):
        raise RuntimeError("MPS is not available")
    source = _embed_headers(
        [l + "\n" for l in source.split("\n")],
        [Path(__file__).parent.parent / "include"],
        set(),
    )
    return torch._C._mps_compileShader(source)


def load_metallib(source):
    # [FUNCTION DOC] Loads a precompiled Metal library from raw bytes (bytes or bytearray) or from
    # a file path (str or path-like) and returns a shader library object. Raises RuntimeError if
    # the matching native loader is missing, TypeError for other source types.
    # KNOWN ISSUE: the bytes path looks up a native name spelled with three l letters
    # (_mps_loadMetalllib), unlike the path variant (_mps_loadMetallibFromPath); verify it is intended.
    r"""Loads a precompiled Metal library (.metallib) and returns a shader
    library object that allows invoking kernels defined in it.

    Args:
        source: Either raw metallib bytes (``bytes``/``bytearray``) or a
            filesystem path (``str``/``os.PathLike``) to a ``.metallib`` file.

    This is useful for loading Metal libraries compiled ahead of time or
    generated by external tools (e.g. Triton, MetalASM).

    Example::

        >>> # xdoctest: +SKIP("requires external .metallib file")
        >>> lib = torch.mps.load_metallib("kernels.metallib")
        >>> x = torch.ones(16, device="mps")
        >>> lib.square(x)
    """
    import os

    if isinstance(source, (bytes, bytearray)):
        if not hasattr(torch._C, "_mps_loadMetalllib"):
            raise RuntimeError("MPS is not available")
        return torch._C._mps_loadMetalllib(bytes(source))
    elif isinstance(source, (str, os.PathLike)):
        if not hasattr(torch._C, "_mps_loadMetallibFromPath"):
            raise RuntimeError("MPS is not available")
        return torch._C._mps_loadMetallibFromPath(str(source))
    else:
        raise TypeError(f"expected bytes or path, got {type(source).__name__}")


def is_available() -> bool:
    # [FUNCTION DOC] Returns True if at least one MPS device is available.
    return device_count() > 0


def _host_alias_storage(storage: "torch.UntypedStorage") -> "torch.UntypedStorage":
    # [FUNCTION DOC] HELPER: Returns a CPU storage sharing memory with the shared-mode Metal buffer
    # behind the given MPS storage (zero-copy host access). The alias keeps the source alive.
    # Raises (from the native layer) if the storage is not backed by such a buffer. Callers must
    # synchronize before and after host access; no ordering is enforced here.
    r"""Returns a CPU :class:`torch.UntypedStorage` that aliases the
    host-visible contents of the MTLBuffer backing ``storage``.

    The returned storage shares memory with ``storage``: writes through the
    CPU alias land directly in the MPS-allocated MTLBuffer, avoiding a
    CPU->MPS staging copy. This is intended for advanced interop with bulk
    loaders (e.g. safetensors) that already know how to write into CPU
    memory.

    The alias storage retains a reference to the source MPS storage, so the
    host pointer remains valid for the alias's lifetime even if the original
    tensor is freed.

    Raises an exception if ``storage`` is not backed by a shared-storage
    ``id<MTLBuffer>`` allocated by the MPS allocator.

    .. warning::
        Use with caution. This bypasses the cache-coherence guarantees that
        the higher-level PyTorch APIs (:meth:`torch.Tensor.cpu`,
        :meth:`torch.Tensor.to`, ``copy_``) provide for you, and makes the
        caller responsible for ordering CPU and GPU accesses to the same
        memory. You should **always** call :func:`torch.mps.synchronize`
        both **before** issuing host reads/writes through the alias (to
        drain any in-flight GPU work that may still be touching the
        buffer) and **after** (before launching GPU work that depends on
        the host writes). Failure to do so can produce stale reads,
        torn writes, or data corruption.
    """
    return torch._C._mps_host_alias_storage(storage)


from . import profiler
from .event import Event


__all__ = [
    "compile_shader",
    "load_metallib",
    "device_count",
    "get_rng_state",
    "manual_seed",
    "seed",
    "set_rng_state",
    "synchronize",
    "empty_cache",
    "set_per_process_memory_fraction",
    "current_allocated_memory",
    "driver_allocated_memory",
    "Event",
    "profiler",
    "recommended_max_memory",
    "is_available",
]
