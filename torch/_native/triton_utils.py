# [MODULE DOC] Triton DSL adapter for the native override registry: detects whether triton is installed and a
# sufficient version (3.x with minor >= 6, unless skip-version env is set), and provides DSL-checked register and
# deregister wrappers. Import-time side effect: registers this module with the global DSL registry under the name triton.
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


_TRITON_DSL_NAME = "triton"
_TRITON_REQUIRED_VERSION_MAJOR = 3
_TRITON_MINIMUM_VERSION_MINOR = 6


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    # [FUNCTION DOC] HELPER: Cached check returning (available, version). False/None on CPU-only builds or when triton is
    # missing (logged at info level); otherwise True and the installed version.
    """
    Check if triton is available

    NOTE: must not import at this point
    """
    # Skip all checks if running on CPU-only binary
    if not _cuda.is_built():
        return (False, None)

    deps = [
        ("triton", "triton"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("triton")
    else:
        # info, not warning: see cutedsl_utils._check_runtime_available for
        # rationale (missing optional deps is the common case; surface via
        # TORCH_LOGS=+native_dsl when needed).
        log.info("triton native DSL ops require: `triton` %s", reason)
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    # [FUNCTION DOC] Return True if triton is available at runtime.
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | Version:
    # [FUNCTION DOC] Return the installed triton version, or None if unavailable.
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_sufficient() -> bool:
    # [FUNCTION DOC] HELPER: Cached check that the triton version is major 3 and minor at least 6, or that the version check
    # is skipped by environment setting; logs at info level and returns False otherwise.
    _, version = _check_runtime_available()

    if version is None:
        return False

    # Either exact version, or same major
    major_ok = version.major == _TRITON_REQUIRED_VERSION_MAJOR
    minor_ok = version.minor >= _TRITON_MINIMUM_VERSION_MINOR

    if (major_ok and minor_ok) or check_native_version_skip():
        return True

    log.info(
        "triton version %s is not sufficient (>= (%s.%s.*)); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _TRITON_REQUIRED_VERSION_MAJOR,
        _TRITON_MINIMUM_VERSION_MINOR,
    )
    return False


def deregister_op_overrides() -> None:
    # [FUNCTION DOC] Disable all overrides registered by the triton DSL via the registry filter.
    """
    Deregister all ops through triton
    """
    _deregister_op_overrides_impl(disable_dsl_names=_TRITON_DSL_NAME)


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
    # [FUNCTION DOC] Register an override under the triton DSL name by forwarding to the registry. Silently does nothing if
    # triton is unavailable, native JIT is disabled, or the version is insufficient.
    # KNOWN ISSUE: allow_multiple_override is forwarded but ignored by the registry.
    """
    See torch/_native/registry.py for the underlying implementation
    and arguments. This is a thin, DSL-checking wrapper over
    _register_op_override_impl
    """
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override_impl(
        _TRITON_DSL_NAME,
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
dsl_registry.register_dsl("triton", cast(DSLModuleProtocol, sys.modules[__name__]))
