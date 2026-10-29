# [MODULE DOC] Helion DSL backend adapter for native op overrides. Availability requires a CUDA (non-ROCm)
# build, the helion package plus the package for the chosen HELION_BACKEND (only triton is supported),
# and helion >= a minimum version unless the version check is skipped. Wraps the generic registry functions.
# Import-time side effect: registers this module as the "helion" DSL in dsl_registry.
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


_HELION_DSL_NAME = "helion"
# Minimum supported Helion: the latest release verified with the native DSL
# kernels (older releases carry the same APIs but are untested).
_HELION_MINIMUM_VERSION = Version("1.2.0")
# HELION_BACKEND name -> (pip package, import module); extend to add more,
# e.g. "cute": ("nvidia-cutlass-dsl", "cutlass").
_HELION_BACKENDS: dict[str, tuple[str, str]] = {"triton": ("triton", "triton")}
_DEFAULT_HELION_BACKEND = "triton"


def _chosen_backend() -> str:
    # [FUNCTION DOC] HELPER: Return the Helion lowering backend name from env HELION_BACKEND (default "triton").
    """Selected Helion lowering backend (HELION_BACKEND env, default triton)."""
    import os

    return os.getenv("HELION_BACKEND", _DEFAULT_HELION_BACKEND)


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    # [FUNCTION DOC] HELPER: Cached probe returning (available, version) for Helion. (False, None) on CPU-only/ROCm
    # builds, an unsupported HELION_BACKEND (info log), or missing packages (info log); otherwise (True, helion version).
    if not _cuda.is_built():
        return (False, None)

    import torch

    if torch.version.hip is not None:
        return (False, None)

    backend = _chosen_backend()
    dep = _HELION_BACKENDS.get(backend)
    if dep is None:
        log.info(
            "Helion native DSL ops support backends %s; HELION_BACKEND=%s",
            tuple(_HELION_BACKENDS),
            backend,
        )
        return (False, None)

    reason = _unavailable_reason([("helion", "helion"), dep])
    if reason is not None:
        log.info(
            "Helion native DSL ops require optional packages `helion` and `%s`; %s",
            dep[0],
            reason,
        )
        return (False, None)
    return (True, _available_version("helion"))


def runtime_available() -> bool:
    # [FUNCTION DOC] Return True if Helion and its backend package are present (version is checked at registration time).
    # Package presence only, like triton/cutedsl; version gated in register_op_override.
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> Version | None:
    # [FUNCTION DOC] Return the installed Helion version, or None if unavailable.
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_sufficient() -> bool:
    # [FUNCTION DOC] HELPER: Cached check that the Helion version is at least the minimum (or the skip-version-check
    # env flag is set). Returns False (with info log) if the version is unknown or too old.
    _, version = _check_runtime_available()
    if version is not None and (
        version >= _HELION_MINIMUM_VERSION or check_native_version_skip()
    ):
        return True

    log.info(
        "helion version %s is not sufficient (>= %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _HELION_MINIMUM_VERSION,
    )
    return False


def deregister_op_overrides() -> None:
    # [FUNCTION DOC] Deregister all op overrides registered under the helion DSL name.
    _deregister_op_overrides_impl(disable_dsl_names=_HELION_DSL_NAME)


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
    # [FUNCTION DOC] Thin wrapper over the registry's register_op_override for the helion DSL. Does nothing if Helion is
    # unavailable, native JIT is disabled globally, or the version is insufficient; otherwise forwards all arguments.
    available, _ = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override_impl(
        _HELION_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


dsl_registry.register_dsl(
    _HELION_DSL_NAME, cast(DSLModuleProtocol, sys.modules[__name__])
)
