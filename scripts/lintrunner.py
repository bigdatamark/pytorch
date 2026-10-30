#!/usr/bin/env python3
# [MODULE DOC] Wrapper that runs lintrunner inside the isolated hook virtual environment (the
# one created by the setup hooks script under the git hooks directory), so developers get the
# same tool versions as the pre-push hook. Also exports helpers for repository root discovery
# and for hashing the lintrunner configuration, which other scripts share. The main entry checks
# that lintrunner is installed, re-runs the plugin init when the configuration hash changed,
# runs lintrunner with the command line arguments and exits with its status. Import has no side
# effects.
"""
Wrapper script to run the isolated hook version of lintrunner.

This allows developers to easily run lintrunner (including with -a for auto-fixes)
using the same isolated environment that the pre-push hook uses, without having
to manually activate/deactivate virtual environments.

Usage:
    python scripts/lintrunner.py          # Check mode (same as git push)
    python scripts/lintrunner.py -a       # Auto-fix mode
    python scripts/lintrunner.py --help   # Show lintrunner help

This module also provides shared functionality for lintrunner hash management.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    # [FUNCTION DOC] Returns the repository root by asking git for the top level directory of
    # the current directory. Exits the process with an error message if git reports a failure.
    """Find repository root using git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        sys.exit("❌ Not in a git repository")


def compute_file_hash(path: Path) -> str:
    # [FUNCTION DOC] Returns the SHA256 hex digest of a file, reading it in fixed size chunks.
    """Returns SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_stored_hash(path: Path) -> str | None:
    # [FUNCTION DOC] Returns the stripped text of a stored hash file, or None if the file does
    # not exist or cannot be read (any exception is swallowed).
    if not path.exists():
        return None
    try:
        return path.read_text().strip()
    except Exception:
        return None


# Venv location - change this if the path changes
HOOK_VENV_PATH = ".git/hooks/linter/.venv"


def get_hook_venv_path() -> Path:
    # [FUNCTION DOC] Returns the hook virtual environment path under the repository root (the
    # location is a module constant). Does not check that it exists.
    """Get the path to the hook virtual environment."""
    repo_root = find_repo_root()
    return repo_root / HOOK_VENV_PATH


def find_hook_venv() -> Path:
    # [FUNCTION DOC] Returns the hook virtual environment path; exits the process with a message
    # pointing to the setup hooks script if the directory does not exist.
    """Locate the isolated hook virtual environment."""
    venv_dir = get_hook_venv_path()

    if not venv_dir.exists():
        sys.exit(
            f"❌ Hook virtual environment not found at {venv_dir}\n"
            "   Please set this up by running: python scripts/setup_hooks.py"
        )

    return venv_dir


def check_lintrunner_installed(venv_dir: Path) -> None:
    # [FUNCTION DOC] Uses the uv tool to query whether the lintrunner package exists in the
    # venv's python; exits the process with a message if the query fails, otherwise prints a
    # confirmation.
    """Check if lintrunner is installed in the given venv, exit if not."""
    result = subprocess.run(
        [
            "uv",
            "pip",
            "show",
            "--python",
            str(venv_dir / "bin" / "python"),
            "lintrunner",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        sys.exit(
            "❌ lintrunner is required but was not found in the hook environment. "
            "Please run `python scripts/setup_hooks.py` to reinstall."
        )
    print("✅ lintrunner is already installed")


def run_lintrunner(venv_dir: Path, args: list[str]) -> int:
    # [FUNCTION DOC] Runs the lintrunner executable from the venv with the given arguments and
    # returns its exit status. The child environment is a copy of the current one with the venv
    # bin directory first on PATH and the virtual env variable set. Output goes straight to the
    # terminal.
    """Run lintrunner command in the specified venv and return exit code."""
    # Run lintrunner directly from the venv's bin directory with environment setup
    lintrunner_exe = venv_dir / "bin" / "lintrunner"
    cmd = [str(lintrunner_exe)] + args
    env = os.environ.copy()

    # PATH: Ensures lintrunner can find other tools in the venv (like python, pip, etc.)
    env["PATH"] = str(venv_dir / "bin") + os.pathsep + env.get("PATH", "")
    # VIRTUAL_ENV: Tells tools like pip_init.py that we're in a venv (prevents --user flag issues)
    env["VIRTUAL_ENV"] = str(venv_dir)

    # Note: Progress tends to be slightly garbled due to terminal control sequences,
    # but functionality and final results will be correct
    return subprocess.call(cmd, env=env)


def initialize_lintrunner_if_needed(venv_dir: Path) -> None:
    # [FUNCTION DOC] Re-initializes lintrunner plugins when the configuration changed. Hashes
    # the lintrunner config file in the repository root and compares with the hash stored in the
    # venv; if missing config, prints a warning and returns. On a difference runs the init
    # command, exits the process if it fails, and stores the new hash; otherwise prints that
    # plugins are up to date.
    """Check if lintrunner needs initialization and run init if needed."""
    repo_root = find_repo_root()
    lintrunner_toml_path = repo_root / ".lintrunner.toml"
    initialized_hash_path = venv_dir / ".lintrunner_plugins_hash"

    if not lintrunner_toml_path.exists():
        print("⚠️ No .lintrunner.toml found. Skipping init.")
        return

    current_hash = compute_file_hash(lintrunner_toml_path)
    stored_hash = read_stored_hash(initialized_hash_path)

    if current_hash != stored_hash:
        print("🔁 Running `lintrunner init` …", file=sys.stderr)
        result = run_lintrunner(venv_dir, ["init"])
        if result != 0:
            sys.exit(f"❌ lintrunner init failed")
        initialized_hash_path.write_text(current_hash)
    else:
        print("✅ Lintrunner plugins already initialized and up to date.")


def main() -> None:
    # [FUNCTION DOC] Entry point. Finds the hook venv, exits if its python is missing, ensures
    # lintrunner is installed and initialized, then runs lintrunner with the script's arguments.
    # If it fails without the auto-fix flag, prints a hint to rerun with that flag. Always exits
    # with lintrunner's status; a keyboard interrupt prints a message and exits with status 1.
    """Run lintrunner in the isolated hook environment."""
    venv_dir = find_hook_venv()
    python_exe = venv_dir / "bin" / "python"

    if not python_exe.exists():
        sys.exit(f"❌ Python executable not found at {python_exe}")

    try:
        print(f"🐍 Virtual env being used: {venv_dir}", file=sys.stderr)

        # 1. Ensure lintrunner binary is available in the venv
        check_lintrunner_installed(venv_dir)

        # 2. Check for plugin updates and re-init if needed
        initialize_lintrunner_if_needed(venv_dir)

        # 3. Run lintrunner with any passed arguments and propagate its exit code
        args = sys.argv[1:]
        result = run_lintrunner(venv_dir, args)

        # If lintrunner failed and we're not already in auto-fix mode, suggest the wrapper
        if result != 0 and "-a" not in args:
            print(
                "\n💡 To auto-fix these issues, run: python scripts/lintrunner.py -a",
                file=sys.stderr,
            )

        sys.exit(result)

    except KeyboardInterrupt:
        print("\n  Lintrunner interrupted by user (KeyboardInterrupt)", file=sys.stderr)
        sys.exit(1)  # Tell git push to fail


if __name__ == "__main__":
    main()
