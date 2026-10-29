# mypy: allow-untyped-defs
# [PACKAGE DOC] torch.multiprocessing: drop-in wrapper over the standard multiprocessing module
# that installs shared-memory reducers for tensors and storages (via init_reductions) so they can
# be sent between processes without copying. Re-exports everything from multiprocessing, the
# spawn helpers (spawn, start_processes, contexts and process exception types), and manages the
# CPU tensor sharing strategy (file_descriptor or file_system; only file_system on macOS and
# Windows) through set/get/get_all_sharing_strategies. Import-time side effects: initializes the
# native multiprocessing support, registers the reducers, and on macOS with Python 3.12.2 or
# newer registers an exit hook that disables the resource tracker finalizer.
"""torch.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
"""

import multiprocessing
import sys

import torch

from .reductions import init_reductions


__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__


# This call adds a Linux specific prctl(2) wrapper function to this module.
# See https://github.com/pytorch/pytorch/pull/14391 for more information.
torch._C._multiprocessing_init()


"""Add helper function to spawn N processes and wait for completion of any of
them."""
from .spawn import (
    ENV_VAR_PARALLEL_START,
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
    SpawnContext,
    start_processes,
)


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    # [FUNCTION DOC] Sets the module-level strategy used to share CPU tensors. Raises
    # AssertionError if new_strategy is not in the set of strategies supported on this platform.
    # Side effect: mutates the global current strategy.
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    if new_strategy not in _all_sharing_strategies:
        raise AssertionError(
            f"invalid sharing strategy {new_strategy!r}, "
            f"expected one of {_all_sharing_strategies}"
        )
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    # [FUNCTION DOC] Returns the name of the current CPU tensor sharing strategy.
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    # [FUNCTION DOC] Returns the set of sharing strategy names supported on this platform (the
    # internal set object itself, not a copy).
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


def _set_thread_name(name: str) -> None:
    # [FUNCTION DOC] HELPER: Sets the OS-level name of the calling thread via the native layer.
    """Set the name of the current thread.

    Args:
        name (str): Name of the current thread.
    """
    torch._C._set_thread_name(name)


def _get_thread_name() -> str:
    # [FUNCTION DOC] HELPER: Returns the OS-level name of the calling thread from the native layer.
    """Get the name of the current thread.

    Returns:
        str: Name of the current thread.
    """
    return torch._C._get_thread_name()


init_reductions()

# Leak ResourceTracker at exit for Python-3.12 on MacOS
# See https://github.com/pytorch/pytorch/issues/153050 and
# https://github.com/python/cpython/issues/88887 for more details
from multiprocessing.resource_tracker import ResourceTracker as _RT


if (
    sys.platform == "darwin"
    and sys.version_info >= (3, 12, 2)
    and hasattr(_RT, "__del__")
):
    import atexit

    def _leak_RT_at_exit():
        # [FUNCTION DOC] HELPER: Exit hook that replaces the resource tracker finalizer with a
        # no-op, so the tracker is intentionally leaked at interpreter exit (workaround for a
        # macOS Python 3.12 issue). Mutates the ResourceTracker class.
        def _noop(x):
            # [FUNCTION DOC] HELPER: Does nothing; stands in for the finalizer.
            pass

        _RT.__del__ = _noop  # type: ignore[attr-defined]

    atexit.register(_leak_RT_at_exit)
