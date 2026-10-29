# [MODULE DOC]
# Script that prepares the repository for clang-tidy: updates git submodules, configures cmake
# to produce the compile commands database, and runs ATen and Python code generation.
# Main contents: run_cmd and one function per step, chained by generate_build_files. Run as a
# script it executes all steps. No import-time side effects.
from __future__ import annotations

import os
import subprocess
import sys


def run_cmd(cmd: list[str]) -> None:
    # [FUNCTION DOC] HELPER: Prints and runs the command with captured output, then prints
    # its stdout and stderr. If the exit status is nonzero, prints a failure message and exits
    # the whole process with status 1.
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        capture_output=True,
    )
    stdout, stderr = (
        result.stdout.decode("utf-8").strip(),
        result.stderr.decode("utf-8").strip(),
    )
    print(stdout)
    print(stderr)
    if result.returncode != 0:
        print(f"Failed to run {cmd}")
        sys.exit(1)


def update_submodules() -> None:
    # [FUNCTION DOC] HELPER: Initializes and updates all git submodules recursively.
    run_cmd(["git", "submodule", "update", "--init", "--recursive"])


def gen_compile_commands() -> None:
    # [FUNCTION DOC] Sets process environment variables (NCCL off, precompiled headers on,
    # clang and clang++ as compilers) and runs the cmake configure step for the current
    # directory into the build directory with the Ninja generator. Side effects: environment
    # mutation and files under build; exits the process on cmake failure.
    """Configure cmake to produce build/compile_commands.json for clang-tidy.

    Configure-only invocation; does not run the build step. The repo-level
    cmake/EnvVarForwarding.cmake forwards BUILD_*/USE_* environment
    variables to the corresponding CMake cache variables, so setting them
    in os.environ before this call propagates them through to CMake.
    """
    os.environ["USE_NCCL"] = "0"
    os.environ["USE_PRECOMPILED_HEADERS"] = "1"
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
    run_cmd(["cmake", "-S", ".", "-B", "build", "-G", "Ninja"])


def run_autogen() -> None:
    # [FUNCTION DOC] HELPER: Runs the torchgen generator for ATen sources into the build
    # directory with per-operator headers, then runs the setup_helpers code generation script
    # with the native functions and tags YAML paths and the lazy TorchScript backend enabled.
    # Paths are relative to the current directory; exits the process on failure.
    run_cmd(
        [
            sys.executable,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
            "--per-operator-headers",
        ]
    )

    run_cmd(
        [
            sys.executable,
            "tools/setup_helpers/generate_code.py",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--gen-lazy-ts-backend",
        ]
    )


def generate_build_files() -> None:
    # [FUNCTION DOC] Runs the full preparation sequence: submodule update, cmake configure,
    # then code generation. Must be run from the repository root.
    update_submodules()
    gen_compile_commands()
    run_autogen()


if __name__ == "__main__":
    generate_build_files()
