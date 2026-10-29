#!/usr/bin/env python3
# [MODULE DOC] Command-line script (arguments: input and output directories) that repairs
# macOS arm64 wheels with delocate so they bundle their dependency libraries. Installs the
# pinned delocate release if the tool is missing, then runs it on every wheel in the input
# directory. Main entries: ensure_delocate, main. No import-time side effects.
"""Repair (delocate) macOS arm64 wheels so they are self-contained.

Usage: repair_wheel.py <input_dir> <output_dir>

Mirrors the .ci/manywheel/repair_wheel.py contract (read raw wheels from
input_dir, write finished wheels to output_dir) but uses delocate -- the macOS
analog of auditwheel/patchelf. delocate bundles dependency dylibs into the
wheel and rewrites their install names to @loader_path. It is
interpreter-independent, so it handles any CPython ABI.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Pinned delocate release.
DELOCATE_PIN = "https://github.com/matthew-brett/delocate/archive/refs/tags/0.10.4.zip"


def ensure_delocate() -> None:
    # [FUNCTION DOC] HELPER: If the delocate-wheel executable is not on the path, pip install
    # the pinned delocate archive using the current Python. Raises a called-process error on
    # failure.
    if shutil.which("delocate-wheel") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", DELOCATE_PIN], check=True
        )


def main() -> None:
    # [FUNCTION DOC] Entry point. Ensures delocate is available, creates the output directory,
    # exits with a message if the input directory has no wheels, runs delocate on each wheel
    # writing into the output directory, and prints how many wheels are there afterwards.
    # Raises a called-process error if any repair fails.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    ensure_delocate()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wheels = sorted(args.input_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No wheels found in {args.input_dir}")

    for whl in wheels:
        subprocess.run(
            ["delocate-wheel", "-v", "-w", str(args.output_dir), str(whl)],
            check=True,
        )

    repaired = list(args.output_dir.glob("*.whl"))
    print(f"Delocated {len(repaired)} wheel(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
