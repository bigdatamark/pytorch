# [MODULE DOC] Git helpers for lumen CLI tasks: a progress logger for GitPython
# operations, cloning an external repository at a commit pinned in a file, and reading
# that pin. Requires GitPython at import time.
"""
Git Utility helpers for CLI tasks.
"""

import logging
from pathlib import Path

from cli.lib.common.path_helper import remove_dir
from git import GitCommandError, RemoteProgress, Repo


logger = logging.getLogger(__name__)


class PrintProgress(RemoteProgress):
    # [CLASS DOC] GitPython progress callback that logs progress through the module
    # logger. State: the interval (percent step between messages) and the last
    # percentage logged, used to avoid repeated lines.
    """Simple progress logger for git operations."""

    def __init__(self, interval: int = 5):
        # [FUNCTION DOC] Initialise the base progress object and set the reporting
        # interval in percent and the last reported percentage (starts unset).
        super().__init__()
        self._last_percent = -1
        self._interval = interval

    def update(self, op_code, cur, max=None, message=""):
        # [FUNCTION DOC] Progress callback. When current and max counts are given, logs
        # the percentage and current line only when it changed and is a multiple of the
        # interval; otherwise logs the message text if there is one. Logging only.
        msg = self._cur_line or message
        if max and cur:
            percent = int(cur / max * 100)
            if percent != self._last_percent and percent % self._interval == 0:
                self._last_percent = percent
                logger.info("Progress: %d%% - %s", percent, msg)
        elif msg:
            logger.info(msg)


def clone_external_repo(target: str, repo: str, dst: str = "", update_submodules=False):
    # [FUNCTION DOC] Clone repo into dst (default: the target name), deleting any
    # existing dst first. Fetches all remotes and tags, checks out the commit pinned
    # for target (see get_post_build_pinned_commit) and optionally updates submodules
    # recursively. Returns the repository object and the commit. Logs and re-raises
    # GitCommandError; a missing pin file raises FileNotFoundError.
    """Clone repository with pinned commit and optional submodules."""
    dst = dst or target

    try:
        logger.info("Cloning %s to %s", target, dst)

        # Clone and fetch
        remove_dir(dst)
        r = Repo.clone_from(repo, dst, progress=PrintProgress())
        r.git.fetch("--all", "--tags", "--force")

        # Checkout pinned commit
        commit = get_post_build_pinned_commit(target)
        logger.info("Checking out pinned %s commit %s", target, commit)
        r.git.checkout(commit)

        # Update submodules if requested
        if update_submodules and r.submodules:
            logger.info("Updating %d submodule(s)", len(r.submodules))
            for sm in r.submodules:
                sm.update(init=True, recursive=True, progress=PrintProgress())

        logger.info("Successfully cloned %s", target)
        return r, commit

    except GitCommandError:
        logger.exception("Git operation failed")
        raise


def get_post_build_pinned_commit(name: str, prefix=".github/ci_commit_pins") -> str:
    # [FUNCTION DOC] Return the stripped text of the pin file named after name under
    # prefix (relative to the current directory). Raises FileNotFoundError if absent.
    path = Path(prefix) / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Pin file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
