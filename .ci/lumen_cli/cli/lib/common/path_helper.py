# [MODULE DOC] Filesystem path helpers for lumen CLI tasks: normalising paths, creating,
# removing and recreating directories, copying files or trees, and existence checks.
"""Path utility helpers for CLI tasks."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)


def get_path(path: str | Path, resolve: bool = False) -> Path:
    # [FUNCTION DOC] Convert a string or Path to a Path (resolved when requested).
    # Raises ValueError for an empty or None value.
    """Convert to Path object, optionally resolving to absolute path."""
    if not path:
        raise ValueError("Path cannot be None or empty")
    result = Path(path)
    return result.resolve() if resolve else result


def ensure_dir_exists(path: str | Path) -> Path:
    # [FUNCTION DOC] Create the directory with all parents if missing and return it as
    # a Path. Raises ValueError for an empty path.
    """Create directory if it doesn't exist."""
    path_obj = get_path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def remove_dir(path: str | Path | None) -> None:
    # [FUNCTION DOC] Recursively delete the path if it exists; an empty or None path is
    # a no-op. Deletes data. KNOWN ISSUE: if the path is a plain file, rmtree raises.
    """Remove directory if it exists."""
    if not path:
        return
    path_obj = get_path(path)
    if path_obj.exists():
        shutil.rmtree(path_obj)


def force_create_dir(path: str | Path) -> Path:
    # [FUNCTION DOC] Delete the directory if present, then create it empty; returns the
    # new directory path. Destroys previous contents.
    """Remove directory if exists, then create fresh empty directory."""
    remove_dir(path)
    return ensure_dir_exists(path)


def copy(src: str | Path, dst: str | Path) -> None:
    # [FUNCTION DOC] Copy a file (keeping metadata) or a directory tree (merging into an
    # existing destination) to dst, creating parent directories. Both paths are
    # resolved first. Raises FileNotFoundError if src is missing and ValueError for
    # other path kinds.
    """Copy file or directory from src to dst."""
    src_path = get_path(src, resolve=True)
    dst_path = get_path(dst, resolve=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source does not exist: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        shutil.copy2(src_path, dst_path)
    elif src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        raise ValueError(f"Unsupported path type: {src_path}")


def is_path_exist(path: str | Path | None) -> bool:
    # [FUNCTION DOC] True when the path is non-empty and exists; False for empty or None.
    """Check if path exists."""
    return bool(path and get_path(path).exists())
