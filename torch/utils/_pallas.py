# [MODULE DOC] Cached feature checks for the optional JAX/Pallas stack (JAX package, Pallas,
# JAX version, CUDA and TPU backends, torch_tpu) and combined checks for Pallas on CPU, CUDA,
# TPU. Every check swallows import or backend errors and reports False. No import-time effects.
import functools

import torch


@functools.cache
def has_jax_package() -> bool:
    # [FUNCTION DOC] Returns True if jax can be imported, else False. Cached; imports jax.
    """Check if JAX is installed."""
    try:
        import jax  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


@functools.cache
def has_pallas_package() -> bool:
    # [FUNCTION DOC] True if jax is installed and its experimental pallas module imports, else
    # False. Cached.
    """Check if Pallas (JAX experimental) is available."""
    if not has_jax_package():
        return False
    try:
        from jax.experimental import (  # noqa: F401  # type: ignore[import-not-found]
            pallas as pl,
        )

        return True
    except ImportError:
        return False


@functools.cache
def get_jax_version(fallback: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    # [FUNCTION DOC] Returns the installed jax version as (major, minor, patch) integers parsed
    # from the first three dot-separated parts, or fallback if jax is missing or the version
    # string cannot be parsed. Cached per fallback value.
    """Get JAX version as (major, minor, patch) tuple."""
    try:
        import jax  # type: ignore[import-not-found]

        version_parts = jax.__version__.split(".")
        major, minor, patch = (int(v) for v in version_parts[:3])
        return (major, minor, patch)
    except (ImportError, ValueError, AttributeError):
        return fallback


@functools.cache
def has_jax_cuda_backend() -> bool:
    # [FUNCTION DOC] True if jax is installed, reports at least one GPU device, and (when torch
    # CUDA is available) the torch device capability major version is 9 or higher. Any exception
    # gives False. Cached.
    """Check if JAX has CUDA backend support with SM90+ (required by Mosaic GPU)."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if CUDA backend is available
        devices = jax.devices("gpu")
        if len(devices) == 0:
            return False

        # Mosaic GPU requires SM90+ (compute capability 9.0+)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major < 9:
                return False

        return True
    except Exception:
        return False


@functools.cache
def has_jax_tpu_backend() -> bool:
    # [FUNCTION DOC] True if jax is installed and reports at least one TPU device; any exception
    # gives False. Cached.
    """Check if JAX has TPU backend support."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if TPU backend is available
        devices = jax.devices("tpu")
        return len(devices) > 0
    except Exception:
        return False


@functools.cache
def has_torch_tpu() -> bool:
    # [FUNCTION DOC] True if the torch package has an attribute for the autoloaded tpu backend.
    # Cached.
    """Check if torch_tpu is installed and available."""
    # The TPU module is an autoloaded out-of-tree backend.
    tpu_module = getattr(torch, "tpu", None)
    return tpu_module is not None


@functools.cache
def has_cpu_pallas() -> bool:
    # [FUNCTION DOC] Same as the Pallas package check (no extra CPU requirement). Cached.
    """Checks for a full Pallas-on-CPU environment."""
    return has_pallas_package()


@functools.cache
def has_cuda_pallas() -> bool:
    # [FUNCTION DOC] True if Pallas is available, torch CUDA is available and the jax CUDA
    # backend check passes. Cached.
    """Checks for a full Pallas-on-CUDA environment."""
    return has_pallas_package() and torch.cuda.is_available() and has_jax_cuda_backend()


@functools.cache
def has_tpu_pallas() -> bool:
    # [FUNCTION DOC] True if Pallas is available, jax has a TPU backend and torch_tpu is present.
    # Cached.
    """Checks for a full Pallas-on-TPU environment."""
    return has_pallas_package() and has_jax_tpu_backend() and has_torch_tpu()


@functools.cache
def has_pallas() -> bool:
    # [FUNCTION DOC] True if any of the CPU, CUDA or TPU Pallas environments is usable. Cached.
    # KNOWN ISSUE: the docstring requires a CUDA or TPU backend, but the CPU check alone (only
    # the package) already makes this True, so the docstring requirement is not enforced.
    """
    Check if Pallas backend is fully available for use.

    Requirements:
    - JAX package installed
    - Pallas (jax.experimental.pallas) available
    - A compatible backend (CUDA or TPU) is available in both PyTorch and JAX.
    """
    return has_cpu_pallas() or has_cuda_pallas() or has_tpu_pallas()
