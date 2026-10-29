# [MODULE DOC] Helpers for locating Inductor on-disk cache directories (general cache, per-user
# default, Triton cache) and a context manager to temporarily override the cache directory.
# Kept free of torch dependencies at import time. No import-time side effects.
import getpass
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

from torch._environment import is_fbcode


# Factoring out to file without torch dependencies


def cache_dir() -> str:
    # [FUNCTION DOC] Returns the absolute Inductor cache directory: the TORCHINDUCTOR_CACHE_DIR
    # environment variable if set, else the per-user default. Side effects: writes the resolved
    # absolute path back into the environment variable and creates the directory if missing.
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        cache_dir = default_cache_dir()
    cache_dir = os.path.abspath(cache_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_cache_dir() -> str:
    # [FUNCTION DOC] Returns the default per-user cache path: a directory named torchinductor_<user>
    # under the system temp dir (or /var/tmp in fbcode). The username comes from getpass, falling
    # back to a uid-based or "unknown_user" name, and path-unsafe characters are replaced by underscores.
    try:
        username = getpass.getuser()
    except (KeyError, ModuleNotFoundError, OSError):
        getuid = getattr(os, "getuid", None)
        username = f"uid_{getuid()}" if callable(getuid) else "unknown_user"
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", username)
    return os.path.join(
        tempfile.gettempdir() if not is_fbcode() else "/var/tmp",
        "torchinductor_" + sanitized_username,
    )


def triton_cache_dir(device: int) -> str:
    # [FUNCTION DOC] Returns the Triton cache directory for a device index: TRITON_CACHE_DIR if set,
    # otherwise a "triton/<device>" subdirectory of the Inductor cache dir (which creates it).
    if (directory := os.getenv("TRITON_CACHE_DIR")) is not None:
        return directory
    return os.path.join(
        cache_dir(),
        "triton",
        str(device),
    )


@contextmanager
def temporary_cache_dir(directory: str) -> Generator[None, None, None]:
    # [FUNCTION DOC] Context manager that points TORCHINDUCTOR_CACHE_DIR at the given directory for
    # the duration of the block. In-memory Inductor caches are cleared on entry and on exit, and the
    # previous environment value is restored (or removed if it was unset) even on error.
    from torch._inductor.utils import clear_caches

    original = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = directory
    try:
        clear_caches()
        yield
    finally:
        clear_caches()
        if original is None:
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = original
