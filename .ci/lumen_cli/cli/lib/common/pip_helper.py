# [MODULE DOC] Helpers that install Python packages with pip or uv, run the current
# interpreter, test whether a package is installed, and pick a wheel file by glob.
# Commands run through run_command; no import-time side effects.
from __future__ import annotations

import glob
import logging
import shlex
import shutil
import sys
from collections.abc import Iterable  # noqa: TC003
from importlib.metadata import PackageNotFoundError, version

from cli.lib.common.utils import run_command


logger = logging.getLogger(__name__)


def pip_install_packages(
    packages: Iterable[str] = (),
    env=None,
    *,
    requirements: str | None = None,
    constraints: str | None = None,
    prefer_uv: bool = False,
) -> None:
    # [FUNCTION DOC] Run an install command with the current interpreter, using uv if
    # prefer_uv is set and uv is on the path, else pip. Adds optional requirements and
    # constraints files and the package names, logs the quoted command and runs it with
    # the optional extra environment. Raises CalledProcessError on failure.
    use_uv = prefer_uv and shutil.which("uv") is not None
    base = (
        [sys.executable, "-m", "uv", "pip", "install"]
        if use_uv
        else [sys.executable, "-m", "pip", "install"]
    )
    cmd = base[:]
    if requirements:
        cmd += ["-r", requirements]
    if constraints:
        cmd += ["-c", constraints]
    cmd += list(packages)
    logger.info("pip installing packages: %s", " ".join(map(shlex.quote, cmd)))
    run_command(" ".join(map(shlex.quote, cmd)), env=env)


def pip_install_first_match(pattern: str, extras: str | None = None, pref_uv=False):
    # [FUNCTION DOC] Install the first (alphabetically) file matching the glob pattern,
    # optionally with an extras suffix. Raises FileNotFoundError if nothing matches.
    wheel = first_matching_pkg(pattern)
    target = f"{wheel}[{extras}]" if extras else wheel
    logger.info("Installing %s...", target)
    pip_install_packages([target], prefer_uv=pref_uv)


def run_python(args: str | list[str], env=None):
    # [FUNCTION DOC] Run the current interpreter with args (a string is split shell
    # style) and optional extra environment; raises CalledProcessError on failure.
    """
    Run the python in the current environment.
    """
    if isinstance(args, str):
        args = shlex.split(args)
    cmd = [sys.executable] + args
    run_command(" ".join(map(shlex.quote, cmd)), env=env)


def pkg_exists(name: str) -> bool:
    # [FUNCTION DOC] Report (and log) whether the distribution is installed, by looking
    # up its metadata version; returns True or False.
    try:
        pkg_version = version(name)
        logger.info("%s already exist with version: %s", name, pkg_version)
        return True
    except PackageNotFoundError:
        logger.info("%s is not installed", name)
        return False


def first_matching_pkg(pattern: str) -> str:
    # [FUNCTION DOC] Return the alphabetically first path matching the glob pattern;
    # raises FileNotFoundError when there is no match.
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No wheel matching: {pattern}")
    return matches[0]
