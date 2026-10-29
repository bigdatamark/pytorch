# [MODULE DOC] Feature detection and small helpers for the optional Triton package: whether it
# is installed, its version, TMA (tensor memory accelerator) API availability, device support,
# backend creation and cache hashing. Most queries are memoized. Imports only standard modules
# at load time.
import functools
import hashlib
import os
from typing import Any


_FAILED_TO_MAP_SEGMENT_FROM_SHARED_OBJECT = "failed to map segment from shared object"


def _triton_cache_dir_for_error_message() -> str | None:
    # [FUNCTION DOC] HELPER: Find the Triton cache directory for error messages: the
    # TRITON_CACHE_DIR environment variable if set, else the Triton knob value, else None if
    # unavailable.
    if triton_cache_dir := os.environ.get("TRITON_CACHE_DIR"):
        return triton_cache_dir

    try:
        from triton.runtime.cache import knobs

        return knobs.cache.dir
    except (AttributeError, ImportError):
        return None


def _raise_triton_cache_load_error(exc: BaseException) -> None:
    # [FUNCTION DOC] HELPER: Raise ImportError chaining exc, with a message explaining that
    # the Triton cache directory may be on a noexec filesystem and suggesting
    # TRITON_CACHE_DIR. Always raises.
    cache_dir = _triton_cache_dir_for_error_message()
    if "TRITON_CACHE_DIR" in os.environ:
        cache_dir_msg = f" (TRITON_CACHE_DIR={cache_dir})"
    else:
        cache_dir_msg = f" ({cache_dir})" if cache_dir else ""
    raise ImportError(
        f"{exc}. This usually means Triton's cache directory{cache_dir_msg} is on "
        "a filesystem mounted with noexec, so the dynamic loader cannot map "
        "generated shared objects. Set TRITON_CACHE_DIR to a directory on an "
        "executable filesystem."
    ) from exc


@functools.cache
def has_triton_package() -> bool:
    # [FUNCTION DOC] Return True if the triton package can be imported (memoized).
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


@functools.cache
def get_triton_version(fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    # [FUNCTION DOC] Return the Triton major and minor version as a pair of ints, or fallback
    # if Triton is not importable (memoized).
    # KNOWN ISSUE: only ImportError is caught; a version string with a non-numeric major or
    #   minor part would raise ValueError.
    try:
        import triton

        major, minor = tuple(int(v) for v in triton.__version__.split(".")[:2])
        return (major, minor)
    except ImportError:
        return fallback


@functools.cache
def _device_supports_tensor_descriptor() -> bool:
    # [FUNCTION DOC] HELPER: Return True if the current device can use tensor descriptors: a
    # CUDA device with capability 9.0 or higher (not HIP), or a Triton CPU backend (memoized).
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and not torch.version.hip
    ) or has_triton_cpu_backend()


@functools.cache
def has_triton_cpu_backend() -> bool:
    # [FUNCTION DOC] Return True if Triton is installed and has a backend named cpu
    # (memoized).
    if has_triton_package():
        import triton

        return "cpu" in triton.backends.backends

    return False


@functools.cache
def has_triton_experimental_host_tma() -> bool:
    # [FUNCTION DOC] Return True if the experimental host TMA descriptor API is usable: Triton
    # present, device supports descriptors, and the experimental descriptor module imports
    # (honoring its enable hook if present). Memoized.
    if has_triton_package():
        if _device_supports_tensor_descriptor():
            try:
                from triton.tools.experimental_descriptor import (  # noqa: F401
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                try:
                    from triton.tools.experimental_descriptor import enable_in_pytorch

                    return enable_in_pytorch()
                except ImportError:
                    return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tensor_descriptor_host_tma() -> bool:
    # [FUNCTION DOC] Return True if Triton's host tensor descriptor API is importable and the
    # device supports descriptors (memoized).
    if has_triton_package():
        if _device_supports_tensor_descriptor():
            try:
                from triton.tools.tensor_descriptor import (  # noqa: F401
                    TensorDescriptor,
                )

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_triton_tma() -> bool:
    # [FUNCTION DOC] Return True if either the tensor descriptor or the experimental host TMA
    # API is available (memoized).
    return has_triton_tensor_descriptor_host_tma() or has_triton_experimental_host_tma()


@functools.cache
def has_triton_tma_device() -> bool:
    # [FUNCTION DOC] Return True if on-device TMA is available: Triton present, a capable
    # CUDA, XPU or CPU backend exists, and either the old tensormap-create API or the new
    # make_tensor_descriptor API imports (memoized).
    if has_triton_package():
        import torch

        if (
            (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (9, 0)
                and not torch.version.hip
            )
            or torch.xpu.is_available()
            or has_triton_cpu_backend()
        ):
            # old API
            try:
                from triton.language.extra.cuda import (  # noqa: F401
                    experimental_device_tensormap_create1d,
                    experimental_device_tensormap_create2d,
                )

                return True
            except ImportError:
                pass

            # new API
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass

    return False


@functools.cache
def has_datacenter_blackwell_tma_device() -> bool:
    # [FUNCTION DOC] Return True only on datacenter Blackwell GPUs (CUDA capability 10.x, not
    # HIP) that also have device and host tensor descriptor support (memoized).
    import torch

    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (10, 0)
        and torch.cuda.get_device_capability() < (11, 0)
        and not torch.version.hip
    ):
        return has_triton_tma_device() and has_triton_tensor_descriptor_host_tma()

    return False


@functools.lru_cache(None)
def has_triton_stable_tma_api() -> bool:
    # [FUNCTION DOC] Return True if Triton's stable TMA API (make_tensor_descriptor) is
    # importable on a capable CUDA, XPU or CPU backend (memoized with an unbounded cache).
    if has_triton_package():
        import torch

        if (
            (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (9, 0)
                and not torch.version.hip
            )
            or torch.xpu.is_available()
            or has_triton_cpu_backend()
        ):
            try:
                from triton.language import make_tensor_descriptor  # noqa: F401

                return True
            except ImportError:
                pass
    return False


@functools.cache
def has_triton_reduction_ordering() -> bool:
    # [FUNCTION DOC] Return True if the installed Triton exposes a ReductionOrdering with an
    # inner-tree option (memoized).
    """Whether the available Triton exposes inner-tree reduction ordering."""
    if has_triton_package():
        try:
            from triton.language import ReductionOrdering

            return hasattr(ReductionOrdering, "INNER_TREE")
        except ImportError:
            pass
    return False


@functools.cache
def has_triton() -> bool:
    # [FUNCTION DOC] Return True if Triton is installed, device detection is not disabled in
    # the inductor config, and some registered non-indexed device is available, Triton capable
    # and has its Triton backend built (memoized). Skips devices whose backend check raises
    # TritonUnavailableError; other errors propagate.
    if not has_triton_package():
        return False

    from torch._inductor.config import triton_disable_device_detection

    if triton_disable_device_detection:
        return False

    from torch._dynamo.device_interface import get_registered_device_interfaces
    from torch._dynamo.exc import TritonUnavailableError

    # A device supports Triton if it is available, reports Triton capability, and
    # its Triton backend is actually built. Capability is gated first so that
    # raise_if_triton_unavailable() only surfaces missing-backend errors (and
    # not, e.g., CUDA's GPUTooOldForTriton for sub-capable devices). We catch the
    # specific TritonUnavailableError rather than RuntimeError so unexpected
    # errors are not silently swallowed.
    for name, device_interface in get_registered_device_interfaces():
        if ":" in name:
            continue
        if not (
            device_interface.is_available() and device_interface.is_triton_capable()
        ):
            continue
        try:
            device_interface.raise_if_triton_unavailable()
        except TritonUnavailableError:
            continue
        return True
    return False


@functools.cache
def triton_backend() -> Any:
    # [FUNCTION DOC] Create and return the Triton backend object for the currently active
    # driver target (memoized). If loading fails with a shared-object mapping error, raises
    # the cache-directory ImportError; other errors propagate.
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    try:
        target = driver.active.get_current_target()
        return make_backend(target)
    except (ImportError, OSError) as e:
        if _FAILED_TO_MAP_SEGMENT_FROM_SHARED_OBJECT in str(e):
            _raise_triton_cache_load_error(e)
        raise


def _extern_libs_key(backend: Any) -> str:
    # [FUNCTION DOC] HELPER: Return a cache-key string built from the SHA-256 digests of the
    # backend's extern library files that exist on disk (empty string if none).
    """Return a cache key fragment for extern libs (e.g. libdevice.10.bc).

    These files affect codegen but are not covered by triton_key() (Python
    sources only) or backend.hash() (ptxas version and arch only).
    """
    opts = backend.parse_options({})
    extern_libs = getattr(opts, "extern_libs", None)
    if not extern_libs:
        return ""
    parts = []
    for name, path in sorted(extern_libs):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                parts.append(f"{name}-{hashlib.sha256(f.read()).hexdigest()}")
    return "-".join(parts)


@functools.cache
def triton_hash_with_backend() -> str:
    # [FUNCTION DOC] Return an upper-case SHA-256 hex digest combining the Triton key and the
    # backend hash (memoized). Creates the backend as a side effect.
    from torch._inductor.runtime.triton_compat import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()
