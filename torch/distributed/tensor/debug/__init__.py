# mypy: allow-untyped-defs
# [PACKAGE DOC] Debugging utilities for DTensor. Exports CommDebugMode (tracks collective
# communication counts and module-level traces) and visualize_sharding (prints a table of how
# a DTensor is laid out across ranks). Also holds private helpers to inspect and clear the
# sharding propagation caches (Python and C++ fast path) and to reset the dispatch logger.
# Import side effect: sets the reported module name of the two exported objects to this package.
import torch._C
from torch.distributed.tensor.debug._comm_mode import CommDebugMode
from torch.distributed.tensor.debug._visualize_sharding import visualize_sharding


__all__ = ["CommDebugMode", "visualize_sharding"]


def _get_python_sharding_prop_cache_info():
    # [FUNCTION DOC] HELPER: Returns the cache statistics (hits, misses, maxsize, current size) of
    # the Python sharding propagation cache of the global DTensor op dispatcher. Debug only; the
    # C++ fast path cache is tracked separately. Takes no arguments, has no side effects.
    """
    Get the cache info for the Python sharding propagation cache, used for debugging purpose only.
    This would return a named tuple showing hits, misses, maxsize and cursize of the sharding
    propagator cache. Note that directly calling into the sharding propagator does not share cache
    state with the DTensor dispatch fast path!
    """
    from torch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined]
    )


def _get_fast_path_sharding_prop_cache_stats():
    # [FUNCTION DOC] HELPER: Returns the (hits, misses) pair of the C++ fast path sharding
    # propagation cache by querying the torch C extension. Debug only; read-only.
    """
    Get a tuple (hits, misses) for the fast path sharding propagation cache, used for debugging
    only.
    """
    return torch._C._get_DTensor_sharding_propagator_cache_stats()


def _clear_python_sharding_prop_cache():
    # [FUNCTION DOC] HELPER: Empties the Python sharding propagation cache of the global DTensor
    # op dispatcher (mutates global cache state). Does not touch the C++ fast path cache.
    """
    Clears the cache for the Python sharding propagation cache, used for debugging purpose only.
    """
    from torch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()  # type:ignore[attr-defined]
    )


def _clear_fast_path_sharding_prop_cache():
    # [FUNCTION DOC] HELPER: Empties the C++ fast path sharding propagation cache via the torch
    # C extension (mutates global state). Does not touch the Python cache.
    """
    Clears the cache for the fast path sharding propagation cache, used for debugging purpose only.
    """
    torch._C._clear_DTensor_sharding_propagator_cache()


def _clear_sharding_prop_cache():
    # [FUNCTION DOC] HELPER: Clears both the Python and the C++ fast path sharding propagation
    # caches by calling the two dedicated clearing helpers in order. Mutates global state.
    """
    Clears both the Python and fast path sharding propagation caches, used for debugging purpose only.
    This is the recommended way to clear all sharding propagation caches.
    """
    _clear_python_sharding_prop_cache()
    _clear_fast_path_sharding_prop_cache()


def _reinit_dispatch_logger():
    # [FUNCTION DOC] HELPER: Tells the C++ extension to reinitialize its cached DTensor dispatch
    # logger state so the next dispatch re-checks whether debug logging is enabled. Mutates
    # global state; call after changing the dispatch logger level.
    """
    Resets the cached DTensor dispatch logger state so that the next DTensor
    dispatch re-checks whether debug logging is enabled. Call this after
    changing the log level on the ``torch.distributed.tensor._dispatch`` logger.
    """
    torch._C._reinit_DTensor_dispatch_logger()


# Set namespace for exposed private names
CommDebugMode.__module__ = "torch.distributed.tensor.debug"
visualize_sharding.__module__ = "torch.distributed.tensor.debug"
