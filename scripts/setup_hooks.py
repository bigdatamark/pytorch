#!/usr/bin/env python3
# [MODULE DOC] Script that installs a git pre-push hook running lintrunner in an isolated
# virtual environment. It runs entirely at import time (there is no main guard): on Windows it
# prints a notice and exits with success; otherwise it requires uv on PATH (exits if absent),
# deletes and recreates the hook venv under the git hooks directory with a pinned Python
# version, installs lintrunner into it, then writes an executable pre-push shell script that
# calls the lintrunner wrapper script with the venv python (skipping quietly if the wrapper is
# missing on an older commit). Temporarily adds the scripts directory to the import path to
# reuse repository root and venv path helpers from the lintrunner module, then removes it.
"""
Bootstrap Git pre‑push hook with isolated virtual environment.

✓ Requires uv to be installed (fails if not available)
✓ Creates isolated venv in .git/hooks/linter/.venv/ for hook dependencies
✓ Installs lintrunner only in the isolated environment
✓ Creates direct git hook that bypasses pre-commit

Run this from the repo root (inside or outside any project venv):

    python scripts/setup_hooks.py

IMPORTANT: The generated git hook references scripts/lintrunner.py. If users checkout
branches that don't have this file, git push will fail with "No such file or directory".
Users would need to either:
1. Re-run the old setup_hooks.py from that branch, or
2. Manually delete .git/hooks/pre-push to disable hooks temporarily, or
3. Switch back to a branch with the new scripts/lintrunner.py
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path


# Add scripts directory to Python path so we can import lintrunner module
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

# Import shared functions from lintrunner module
from lintrunner import find_repo_root, get_hook_venv_path


# Restore sys.path to avoid affecting other imports
sys.path.pop(0)


# ───────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────
def run(cmd: list[str], cwd: Path = None) -> None:
    # [FUNCTION DOC] HELPER: Echoes the command line to stdout and runs it, raising an error on
    # a nonzero exit status. Optional working directory.
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def which(cmd: str) -> bool:
    # [FUNCTION DOC] HELPER: Returns True if the named executable is found on PATH.
    return shutil.which(cmd) is not None


def ensure_uv() -> None:
    # [FUNCTION DOC] HELPER: Returns silently if uv is available; otherwise exits the process
    # with a message telling the user to install uv and rerun this script.
    if which("uv"):
        return

    sys.exit(
        "\n❌  uv is required but was not found on your PATH.\n"
        "    Please install uv first using the instructions at:\n"
        "    https://docs.astral.sh/uv/getting-started/installation/\n"
        "    Then rerun  python scripts/setup_hooks.py\n"
    )


if sys.platform.startswith("win"):
    print(
        "\n⚠️  Lintrunner is not supported on Windows, so there are no pre-push hooks to add. Exiting setup.\n"
    )
    sys.exit(0)

# ───────────────────────────────────────────
# 1. Setup isolated hook environment
# ───────────────────────────────────────────

ensure_uv()

# Find repo root and setup hook directory
repo_root = find_repo_root()
venv_dir = get_hook_venv_path()
hooks_dir = venv_dir.parent.parent  # Go from .git/hooks/linter/.venv to .git/hooks


print(f"Setting up isolated hook environment in {venv_dir}")

# Create isolated virtual environment for hooks
if venv_dir.exists():
    print("Removing existing hook venv...")
    shutil.rmtree(venv_dir)

run(["uv", "venv", str(venv_dir), "--python", "3.10"])

# Install lintrunner in the isolated environment
print("Installing lintrunner in isolated environment...")
run(
    ["uv", "pip", "install", "--python", str(venv_dir / "bin" / "python"), "lintrunner"]
)

# ───────────────────────────────────────────
# 2. Create direct git pre-push hook
# ───────────────────────────────────────────

pre_push_hook = hooks_dir / "pre-push"
python_exe = venv_dir / "bin" / "python"
lintrunner_script_path_quoted = shlex.quote(
    str(repo_root / "scripts" / "lintrunner.py")
)

hook_script = f"""#!/bin/bash
set -e

# Check if lintrunner script exists (user might be on older commit)
if [ ! -f {lintrunner_script_path_quoted} ]; then
    echo "⚠️  {lintrunner_script_path_quoted} not found - skipping linting (likely on an older commit)"
    exit 0
fi

# Run lintrunner wrapper using the isolated venv's Python
{shlex.quote(str(python_exe))} {lintrunner_script_path_quoted}
"""

print(f"Creating git pre-push hook at {pre_push_hook}")
pre_push_hook.write_text(hook_script)
pre_push_hook.chmod(0o755)  # Make executable

print(
    "\n✅  Isolated hook environment created and pre‑push hook is active.\n"
    "   Lintrunner will now run automatically on every `git push`.\n"
    f"   Hook dependencies are isolated in {venv_dir}\n"
)
