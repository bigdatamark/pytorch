#!/usr/bin/env python3
# [MODULE DOC] Command-line script (argument: output directory) that builds a macOS arm64
# PyTorch wheel. Copies the environment, defaults the host platform tag and arch flags to
# arm64 (existing values win), installs the build package with pip and runs the build module
# with the wheel and no-isolation options. Main entry: main. No import-time side effects.
"""Build a macOS arm64 PyTorch wheel.

Usage: build_wheel.py <output_dir>

Mirrors .ci/manywheel/build_wheel.py. Build flags are expected in the
environment (set by build_env_setup.py and sourced by build.sh); this script
adds only the macOS host-platform plumbing and runs `python -m build`.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Force the wheel's platform tag and compile arch to arm64 regardless of the
# build host, via `_PYTHON_HOST_PLATFORM` / `ARCHFLAGS`.
MACOS_PLATFORM = "macosx-14.0-arm64"


def main() -> None:
    # [FUNCTION DOC] Entry point. Parses the output directory, prepares the child environment
    # with default platform and arch settings, then runs pip install build and the wheel build
    # as subprocesses. Raises a called-process error if either fails. Writes wheels to the
    # output directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("_PYTHON_HOST_PLATFORM", MACOS_PLATFORM)
    env.setdefault("ARCHFLAGS", "-arch arm64")

    subprocess.run([sys.executable, "-m", "pip", "install", "build"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(args.output_dir),
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
