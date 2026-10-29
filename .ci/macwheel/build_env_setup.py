#!/usr/bin/env python3
# [MODULE DOC] Command-line script run once per Python before the macOS arm64 wheel build.
# Builds a mapping of static macOS build flags (MACOS_BUILD_ENV), adds the OpenMP prefix when
# the staged conda-forge libomp directory exists, and writes the result as shell export lines
# to the file given by --env-out so build.sh can source them. Main entry: main. No import-time
# side effects beyond defining constants.
"""macOS arm64 build environment setup (runs once per Python before the build).

Mirrors .ci/manywheel/build_env_setup.py for the macOS CD wheel build. macOS
needs far less toolchain wiring than the manylinux path: there is no
CUDA/ROCm/XPU split and the heavy deps come from the runner image or Homebrew.
This script just emits the macOS build flags to the --env-out file so the
caller (build.sh) can source them into the wheel build subprocess. Without that
handoff the exports made here die with this process.

Environment variables written (to --env-out):
    OMP_PREFIX - if /opt/llvm-openmp exists it is exported so the build links
                 the conda-forge libomp (supports older macOS than the Homebrew
                 build). See .ci/macwheel/install_libomp.sh.
    (plus the static macOS build flags in MACOS_BUILD_ENV)
"""

import argparse
from pathlib import Path


# macOS arm64 build flags.
# USE_DISTRIBUTED=1 needs libuv, built as part of the tensorpipe submodule.
# MKLDNN is off on Apple silicon.
MACOS_BUILD_ENV: dict[str, str] = {
    "TH_BINARY_BUILD": "1",
    "INSTALL_TEST": "0",
    "MACOSX_DEPLOYMENT_TARGET": "14.0",
    "USE_DISTRIBUTED": "1",
    "USE_MKLDNN": "OFF",
    "BUILD_TEST": "OFF",
    "USE_PYTORCH_METAL_EXPORT": "1",
    "USE_COREML_DELEGATE": "1",
}

OMP_PREFIX = Path("/opt/llvm-openmp")


def shell_quote(value: str) -> str:
    # [FUNCTION DOC] HELPER: Return the value unchanged if it is non-empty and made only of
    # letters, digits and a few safe punctuation characters; otherwise return it wrapped in
    # single quotes with embedded single quotes escaped for a POSIX shell.
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    # [FUNCTION DOC] Write one export line per key/value pair (values shell-quoted), joined by
    # newlines with a trailing newline, to the given path, overwriting it. Does nothing when
    # the path is None. Side effect: file write; raises OS errors if the path is unwritable.
    """Write `export KEY=VALUE` lines for the parent shell to source."""
    if path is None:
        return
    lines = [f"export {k}={shell_quote(v)}" for k, v in env.items()]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses the optional --env-out path, copies the static macOS
    # flags, adds the OpenMP prefix variable if that directory exists, writes the exports to
    # the file (if requested) and prints a confirmation. Returns nothing; exits via argparse
    # on bad arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-out",
        type=Path,
        help="Write `export KEY=VALUE` lines here for build.sh to source.",
    )
    args = parser.parse_args()

    env_out = dict(MACOS_BUILD_ENV)
    # Prefer the conda-forge libomp staged at /opt/llvm-openmp; otherwise the
    # build falls back to the Homebrew libomp installed in build_install_deps.py.
    if OMP_PREFIX.is_dir():
        env_out["OMP_PREFIX"] = str(OMP_PREFIX)

    write_env_exports(env_out, args.env_out)
    print("macOS build environment configured")


if __name__ == "__main__":
    main()
