# [MODULE DOC] CuTeDSL backend adapter for native op overrides. Gates availability on a CUDA (non-ROCm)
# build, the optional packages nvidia-cutlass-dsl and apache-tvm-ffi, and a known-good cutlass DSL version
# set, then wraps the generic registry functions so overrides are only registered when usable.
# Import-time side effect: registers this module itself as the "cutedsl" DSL in dsl_registry.
import functools
import logging
import sys
from typing import cast

from torch._vendor.packaging.version import Version

from ..backends import cuda as _cuda
from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .dsl_registry import dsl_registry, DSLModuleProtocol
from .registry import (
    _OpCondFn,
    _OpImplFn,
    deregister_op_overrides as _deregister_op_overrides_impl,
    register_op_override as _register_op_override_impl,
)


log = logging.getLogger(__name__)


_CUTEDSL_DSL_NAME = "cutedsl"
_CUTEDSL_REQUIRED_VERSIONS: set[Version] = {
    # Current version - Note Version.from_part(release=(4.4.1)) is better
    #                   but > v26 of packaging.
    Version(f"{4}.{4}.{1}"),
    Version(f"{4}.{4}.{2}"),
    Version(f"{4}.{5}.{2}"),
}


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    # [FUNCTION DOC] HELPER: Cached probe returning (available, version) for CuTeDSL. Returns (False, None) on
    # CPU-only or ROCm builds or when required packages are missing (logged at info level); otherwise
    # (True, installed nvidia_cutlass_dsl version). Does not import the DSL packages.
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    # Skip all checks if running on CPU-only binary
    if not _cuda.is_built():
        return (False, None)

    # CuTeDSL kernels require genuine NVIDIA GPUs; on ROCm/HIP builds the
    # CUDA compatibility layer reports is_built()=True but cuInit() will fail
    # because no NVIDIA driver is present.
    import torch

    if torch.version.hip is not None:
        return (False, None)

    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
        ("apache_tvm_ffi", "tvm_ffi"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("nvidia_cutlass_dsl")
    else:
        # info, not warning: missing optional deps is the common case on stock
        # builds and we don't want to spam stderr on `import torch`. Surface
        # it via TORCH_LOGS=+native_dsl when diagnosing why an override is
        # silent.
        log.info(
            "CuTeDSL operators require optional Python packages "
            "`nvidia-cutlass-dsl` and `apache-tvm-ffi`; "
            "%s",
            reason,
        )
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    # [FUNCTION DOC] Return True if CuTeDSL and its dependencies are available in this environment.
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | Version:
    # [FUNCTION DOC] Return the installed CuTeDSL (nvidia_cutlass_dsl) version, or None if unavailable.
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_ok() -> bool:
    # [FUNCTION DOC] HELPER: Cached check that the CuTeDSL version is acceptable. True if the skip-version-check
    # env flag is set or the version is in the known-good set; otherwise logs an info message and returns False.
    _, version = _check_runtime_available()
    if check_native_version_skip() or (version in _CUTEDSL_REQUIRED_VERSIONS):
        return True

    log.info(
        "cutedsl version %s is not known-good (ok: %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _CUTEDSL_REQUIRED_VERSIONS,
    )
    return False


def deregister_op_overrides() -> None:
    # [FUNCTION DOC] Deregister all op overrides that were registered under the cutedsl DSL name via the registry.
    """
    Deregister all ops through cuteDSL
    """
    _deregister_op_overrides_impl(disable_dsl_names=_CUTEDSL_DSL_NAME)


def register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    # [FUNCTION DOC] Thin wrapper over the registry's register_op_override for the cutedsl DSL. Silently does
    # nothing if CuTeDSL is unavailable, native JIT is globally disabled, or the version is not accepted;
    # otherwise forwards all arguments (including allow_multiple_override and unconditional_override).
    """
    See torch/_native/registry.py for the underlying implementation
    and arguments. This is a thin, DSL-checking wrapper over
    _register_op_override_impl
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_ok():
        return

    _register_op_override_impl(
        _CUTEDSL_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


# Register this DSL module with the registry
# Note: Import-time registration ensures DSL is available when module is loaded
dsl_registry.register_dsl("cutedsl", cast(DSLModuleProtocol, sys.modules[__name__]))
