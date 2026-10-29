# [MODULE DOC] HELPER: Shared environment/dependency checks for native DSL op backends.
# Provides cached global kill-switch checks (env vars), a dependency availability probe
# that does not import packages, and an installed-version lookup. No import-time side effects.
import importlib
import importlib.metadata
import os
from functools import cache

from torch._vendor.packaging import version as _packaging_version


@cache
def check_native_jit_disabled() -> bool:
    # [FUNCTION DOC] Cached check of env var TORCH_DISABLE_NATIVE_JIT; returns True only if it equals 1.
    # Used as the single point to globally disable native DSL ops. Raises ValueError if the
    # variable is set to a non-integer string.
    """
    Single point to check if native DSL ops are disabled globally,
    checked via:
    TORCH_DISABLE_NATIVE_JIT=1
    """
    return int(os.getenv("TORCH_DISABLE_NATIVE_JIT", 0)) == 1


def _unavailable_reason(deps: list[tuple[str, str]]) -> None | str:
    # [FUNCTION DOC] HELPER: Given a list of (package_name, module_name) pairs, return None if all
    # modules can be found via find_spec, else a message naming the first missing package.
    # Does not import the modules; missing parents of dotted names are treated as not found.
    """
    Check availability of required packages - cuteDSL & deps,
    informing user what (if anything) is missing

    NOTE: Doesn't actually import anything.
    """
    for package_name, module_name in deps:
        # find_spec raises ModuleNotFoundError for dotted names (e.g.
        # "nvmath.bindings") when the parent package is absent.
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            found = False
        if not found:
            return (
                f"missing optional dependency `{package_name}` "
                f"(importlib.util.find_spec({package_name}) failed)"
            )
    return None


def _available_version(package: str) -> _packaging_version.Version | None:
    # [FUNCTION DOC] HELPER: Return the installed version of a package as a parsed Version object,
    # or None if the package is not installed or its version string cannot be parsed.
    # KNOWN ISSUE: docstring says it returns a (major, minor, patch) tuple, but a packaging Version is returned.
    """
    Get the installed version of a package as (major, minor, patch).

    Handles pre-release suffixes like "0.7.0rc1" or "3.1.0.post1" by
    stripping non-numeric tails from each component. Returns None on
    parse failure.
    """
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None

    try:
        v = _packaging_version.parse(version)
    except _packaging_version.InvalidVersion:
        return None

    return v


@cache
def check_native_version_skip() -> bool:
    # [FUNCTION DOC] Cached check of env var TORCH_NATIVE_SKIP_VERSION_CHECK; True only if it equals 1.
    # Used to bypass version gating of native DSL dependencies. Raises ValueError on a non-integer value.
    """
    Single point to check if native DSL version gating should be skipped,
    checked via:
    TORCH_NATIVE_SKIP_VERSION_CHECK=1
    """
    return int(os.getenv("TORCH_NATIVE_SKIP_VERSION_CHECK", 0)) == 1
