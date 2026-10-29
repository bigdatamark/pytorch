# [MODULE DOC] Runs generated fuzzer programs in a subprocess. Provides a helper that builds the
# child environment (copying the current environment, propagating sys.path as PYTHONPATH, and
# applying the active device plugin's runtime-env hook) and the ProgramRunner class. No
# import-time side effects.
"""
Program runner utilities for PyTorch fuzzer.
This module handles running and testing generated PyTorch programs.
"""

import os
import subprocess
import sys


def _build_subprocess_env(*, exclude_primary_device: bool = False) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Return a copy of the process environment mapping with PYTHONPATH set
    # to the joined non-empty entries of sys.path. If the active device info defines a
    # select_runtime_env hook, the mapping is passed through it along with the
    # exclude_primary_device flag and the hook's result is returned. Imports codegen lazily.
    """Build the subprocess environment, applying the active plugin's hook (if any)."""
    from torchfuzz.codegen import get_device_info

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)

    select_runtime_env = get_device_info().select_runtime_env
    if select_runtime_env is not None:
        env = select_runtime_env(env, exclude_primary_device=exclude_primary_device)
    return env


class ProgramRunner:
    # [CLASS DOC] Stateless runner that executes a generated program file with the current Python
    # interpreter and reports its output. Key method: run_program.
    """Runs generated PyTorch programs and handles output/error reporting."""

    def __init__(self):
        # [FUNCTION DOC] No-op constructor; the runner holds no state.
        pass

    def run_program(self, program_path):
        # [FUNCTION DOC] Run the program at program_path in a child interpreter with captured
        # output, using an environment that excludes the primary device. Prints the path, then
        # stdout and stderr; returns True on success. On a nonzero exit it prints the output, the
        # program source and the exit code, then terminates the whole process with status 1
        # instead of returning False.
        # KNOWN ISSUE: the docstring promises a False return on failure, but failure exits.
        """
        Run a generated Python program and handle output/errors.

        Args:
            program_path: Path to the Python program to run

        Returns:
            bool: True if program ran successfully, False otherwise
        """
        abs_path = os.path.abspath(program_path)
        print(f"Running: {abs_path}")

        env = _build_subprocess_env(exclude_primary_device=True)

        try:
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            print("=== Program Output ===")
            print(result.stdout)
            print(result.stderr)
            return True

        except subprocess.CalledProcessError as e:
            print("=== Program Output (Failure) ===")
            print(e.stdout)
            print(e.stderr)
            print("===============================")
            print("=== Program Source ===")
            with open(abs_path) as f:
                print(f.read())
            print("======================")
            print(f"Program exited with code: {e.returncode}")
            sys.exit(1)
