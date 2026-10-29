"""Dynamic-metadata provider for the dependencies field.

Reads PYTORCH_EXTRA_INSTALL_REQUIREMENTS (pipe-separated PEP 508 dependency
strings) and appends them to the base dependency list.  Also handles
BUILD_PYTHON_ONLY which adds a dependency on the libtorch wheel package.

The dynamic_wheel hook marks dependencies as Dynamic in the sdist PKG-INFO:
the env vars above can legitimately differ between the sdist build and a
wheel built from it (e.g. CUDA nightly wheels).
"""

# [MODULE DOC]
# scikit-build-core dynamic metadata provider for the dependencies field. Exports
# dynamic_metadata (builds the dependency list from a base list plus environment variables)
# and dynamic_wheel (marks the field dynamic for wheels built from an sdist). Defines
# BASE_DEPENDENCIES. Imports the sibling _common module at top level. No other import-time
# side effects.
from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

# _common is resolved at build time via scikit-build-core's provider path,
# not statically importable from the repo root.
from _common import get_torch_version  # pyrefly: ignore[missing-import]


if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["dynamic_metadata", "dynamic_wheel"]

BASE_DEPENDENCIES = [
    "filelock",
    "typing-extensions>=4.10.0",
    "setuptools>=77.0.3",
    "sympy>=1.13.3",
    "networkx>=2.5.1",
    "jinja2",
    "fsspec>=0.8.5",
]


def _is_truthy(val: str | None) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the value is set and, uppercased, is one of ON, 1,
    # YES, TRUE or Y; False for None or anything else.
    return val is not None and val.upper() in ("ON", "1", "YES", "TRUE", "Y")


def dynamic_metadata(
    settings: Mapping[str, Any],
    project: Mapping[str, Any],
) -> dict[str, Any]:
    # [FUNCTION DOC] Provider hook returning a mapping with the dependencies list. Starts from
    # the base dependencies; if BUILD_PYTHON_ONLY is truthy, appends the libtorch wheel package
    # (name from LIBTORCH_PACKAGE_NAME, default torch_no_python) pinned to the torch version;
    # then appends the pipe-separated entries of PYTORCH_EXTRA_INSTALL_REQUIREMENTS, stripped
    # and skipping empty ones. Raises RuntimeError if any settings are passed. The project
    # argument is unused. Reads environment only.
    if settings:
        msg = f"This provider takes no settings, got {sorted(settings)}"
        raise RuntimeError(msg)

    deps = list(BASE_DEPENDENCIES)

    # BUILD_PYTHON_ONLY: add libtorch wheel as a dependency
    if _is_truthy(os.environ.get("BUILD_PYTHON_ONLY")):
        libtorch_pkg = os.environ.get("LIBTORCH_PACKAGE_NAME", "torch_no_python")
        version = get_torch_version()
        deps.append(f"{libtorch_pkg}=={version}")

    # PYTORCH_EXTRA_INSTALL_REQUIREMENTS: pipe-separated PEP 508 strings
    extra = os.environ.get("PYTORCH_EXTRA_INSTALL_REQUIREMENTS")
    if extra:
        deps.extend(r.strip() for r in extra.split("|") if r.strip())

    return {"dependencies": deps}


def dynamic_wheel(settings: Mapping[str, Any]) -> dict[str, bool]:
    # [FUNCTION DOC] Provider hook declaring that the dependencies field is dynamic in the
    # wheel metadata (it may differ from the sdist). Always returns a mapping with that field
    # set true; settings are ignored.
    return {"dependencies": True}
