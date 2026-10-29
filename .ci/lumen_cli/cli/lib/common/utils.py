# [MODULE DOC] General helpers for lumen CLI tasks: running subprocess commands, string
# to boolean conversion, context managers that temporarily change environment variables
# or the working directory, and discovery of wheel files. No import-time side effects.
"""
General Utility helpers for CLI tasks.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


logger = logging.getLogger(__name__)


def run_command(
    cmd: str,
    use_shell: bool = False,
    log_cmd: bool = True,
    cwd: str | None = None,
    env: dict | None = None,
    check: bool = True,
) -> int:
    # [FUNCTION DOC] Run a command string, streaming output to the current stdout and
    # stderr. Without use_shell the string is split shell style and executed directly;
    # with it, it runs under bash. The environment is the current one overlaid with env.
    # Logs the command unless log_cmd is false. Returns the exit code; when check is
    # true a non-zero exit is logged and raises CalledProcessError.
    """Run a command with optional shell execution."""
    if use_shell:
        args = cmd
        log_prefix = "[shell]"
        executable = "/bin/bash"
    else:
        args = shlex.split(cmd)
        log_prefix = "[cmd]"
        executable = None

    if log_cmd:
        display_cmd = cmd if use_shell else " ".join(args)
        logger.info("%s %s", log_prefix, display_cmd)

    run_env = {**os.environ, **(env or {})}

    proc = subprocess.run(
        args,
        shell=use_shell,
        executable=executable,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=cwd,
        env=run_env,
        check=False,
    )

    if check and proc.returncode != 0:
        logger.error(
            "%s Command failed (exit %s): %s", log_prefix, proc.returncode, cmd
        )
        raise subprocess.CalledProcessError(
            proc.returncode, args if not use_shell else cmd
        )

    return proc.returncode


def str2bool(value: str | None) -> bool:
    # [FUNCTION DOC] Parse a truthy or falsy word case-insensitively after trimming.
    # Empty or None gives False. Accepted true words include 1, true, yes, on, enabled
    # and found; false words include 0, false, no, off and disable. Raises ValueError
    # for a non-string value or an unknown word.
    """Convert environment variables to boolean values."""
    if not value:
        return False
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a string value for boolean conversion, got {type(value)}"
        )
    value = value.strip().lower()

    true_value_set = {"1", "true", "t", "yes", "y", "on", "enable", "enabled", "found"}
    false_value_set = {"0", "false", "f", "no", "n", "off", "disable"}

    if value in true_value_set:
        return True
    if value in false_value_set:
        return False
    raise ValueError(f"Invalid string value for boolean conversion: {value}")


@contextmanager
def temp_environ(updates: dict[str, str]):
    # [FUNCTION DOC] Context manager that sets the given variables in the process
    # environment for the duration of the block, then restores prior values or removes
    # variables that did not exist, even if the block raises. Mutates os.environ.
    """
    Temporarily set environment variables and restore them after the block.
    Args:
        updates: Dict of environment variables to set.
    """
    missing = object()
    old: dict[str, str | object] = {k: os.environ.get(k, missing) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v  # type: ignore[arg-type]


@contextmanager
def working_directory(path: str):
    # [FUNCTION DOC] Context manager that changes the process working directory to path
    # for the block and restores the previous one afterwards (even on error). An empty
    # path does nothing. Raises OSError if the directory cannot be entered.
    """
    Temporarily change the working directory inside a context.
    """
    if not path:
        # No-op context
        yield
        return
    prev_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


def get_wheels(
    output_dir: Path,
    max_depth: int | None = None,
) -> list[str]:
    # [FUNCTION DOC] Walk output_dir and collect every file ending in .whl, skipping
    # directories deeper than max_depth when given. Returns a list of mappings with the
    # package name (file name up to the first dash) and the path relative to
    # output_dir; an empty list if the directory is missing.
    # KNOWN ISSUE: the return annotation says list of strings but mappings are returned.
    """Return a list of wheels found in the given output directory."""
    root = Path(output_dir)
    if not root.exists():
        return []
    items = []
    for dirpath, _, filenames in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if max_depth is not None and len(depth) > max_depth:
            continue
        for fname in sorted(filenames):
            if fname.endswith(".whl"):
                pkg = fname.split("-")[0]
                relpath = str((Path(dirpath) / fname).relative_to(root))
                items.append({"pkg": pkg, "relpath": relpath})
    return items
