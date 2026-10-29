# mypy: allow-untyped-defs
# [PACKAGE DOC] Public entry point of torch.distributed. Reports availability of the
# distributed build, initializes the c10d C++ bindings at import time (raises RuntimeError
# on failure), and exposes the error classes (DistError and subclasses), the c10d
# primitives (stores, ProcessGroup, Reducer, comm hooks, debug level), DeviceMesh,
# init_device_mesh, rendezvous helpers and everything from distributed_c10d.
# When the build lacks distributed support only stub ProcessGroup/GroupName are installed.
# Import-time side effects: sets the debug level from the environment.
import contextlib
import logging
import sys
import traceback
import typing
from datetime import timedelta

import torch


RankType = int | torch.SymInt


log = logging.getLogger(__name__)


def is_available() -> bool:
    # [FUNCTION DOC] Returns True if this PyTorch build has the c10d distributed bindings
    # (detected by the presence of the _c10d_init attribute on torch._C). No side effects.
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch._C, "_c10d_init")


_spmd_types_available: bool | None = None


def _is_spmd_types_available() -> bool:
    # [FUNCTION DOC] HELPER: Returns whether the optional spmd_types package can be found.
    # The result is cached in the module-level _spmd_types_available on first call.
    """Check if the spmd_types package is installed."""
    import importlib.util

    global _spmd_types_available
    if _spmd_types_available is None:
        _spmd_types_available = importlib.util.find_spec("spmd_types") is not None
    return _spmd_types_available


def _spmd_no_typecheck():
    # [FUNCTION DOC] HELPER: Returns a context manager that disables spmd_types type
    # checking when spmd_types is installed, otherwise a do-nothing context manager.
    """
    Return a spmd_types no_typecheck context, or a no-op if not installed.
    """
    if _is_spmd_types_available():
        import spmd_types

        return spmd_types.no_typecheck()

    @contextlib.contextmanager
    def no_typecheck():
        # [FUNCTION DOC] HELPER: Fallback context manager that does nothing.
        yield

    return no_typecheck()


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

# Custom Runtime Errors thrown from the distributed package
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError

if is_available():
    from torch._C._distributed_c10d import (
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _make_nccl_premul_sum,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _test_python_store,
        _verify_params_across_processes,
        Backend as _Backend,
        BuiltinCommHookType,
        DebugLevel,
        FileStore,
        get_debug_level,
        GradBucket,
        Logger,
        PrefixStore,
        ProcessGroup as ProcessGroup,
        Reducer,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work as _Work,
    )

    def _make_distributed_pdb():
        # [FUNCTION DOC] HELPER: Builds a debugger instance usable inside multiprocessing
        # children. Its interaction step temporarily replaces stdin with the terminal
        # device file and restores the original stdin afterwards. pdb is imported lazily.
        """
        Supports using PDB from inside a multiprocessing child process.

        Usage:
        _make_distributed_pdb().set_trace()
        """

        # Lazy import pdb only if we set breakpoints.
        import pdb

        class _DistributedPdb(pdb.Pdb):
            # [CLASS DOC] HELPER: Pdb subclass that reads debugger input from the
            # terminal device instead of the (possibly redirected) process stdin.
            def interaction(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Runs the normal pdb interaction with stdin
                # swapped to the terminal device; always restores the previous stdin.
                _stdin = sys.stdin
                try:
                    with open("/dev/stdin") as sys.stdin:
                        pdb.Pdb.interaction(self, *args, **kwargs)
                finally:
                    sys.stdin = _stdin

        return _DistributedPdb()

    _breakpoint_cache: dict[int, typing.Any] = {}

    def breakpoint(rank: int = 0, skip: int = 0, timeout_s=3600):
        # [FUNCTION DOC] Opens a debugger on one rank while the other ranks wait at a barrier.
        # Args: rank is the rank that stops; skip ignores that many earlier hits (counted per
        # call-site key derived from the current exception traceback text, which is a weak
        # key); timeout_s, if not None, sets every known process group's timeout to that many
        # seconds so the debug session is not interrupted. Returns early when skipping.
        # Side effects: mutates the module-level breakpoint counter cache, changes group
        # timeouts, and runs a barrier with meta-dispatch TLS and Python dispatch disabled,
        # restoring the TLS state afterwards.
        """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
            skip (int): Skip the first ``skip`` calls to this breakpoint. Default: ``0``.
        """
        if skip > 0:
            key = hash(str(traceback.format_exc()))
            counter = _breakpoint_cache.get(key, 0) + 1
            _breakpoint_cache[key] = counter
            if counter <= skip:
                log.warning("Skip the breakpoint, counter=%d", counter)
                return

        # avoid having the default timeout (if short) interrupt your debug session
        if timeout_s is not None:
            for group in torch.distributed.distributed_c10d._pg_map:
                torch.distributed.distributed_c10d.set_timeout(
                    timedelta(seconds=timeout_s), group
                )

        if get_rank() == rank:
            pdb = _make_distributed_pdb()
            pdb.message(
                "\n!!! ATTENTION !!!\n\n"
                f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
            )
            pdb.set_trace()
        # If Meta/Python keys are in the TLS, we want to make sure that we ignore them
        # and hit the (default) CPU/CUDA implementation of barrier.
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
        torch._C._set_meta_in_tls_dispatch_include(False)
        try:
            barrier()
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
            del guard

    if sys.platform != "win32":
        from torch._C._distributed_c10d import HashStore

    from .device_mesh import DeviceMesh, init_device_mesh

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import *  # noqa: F403
    from .distributed_c10d import (
        _all_gather_base,
        _coalescing_manager,
        _CoalescingManager,
        _create_process_group_wrapper,
        _get_process_group_name,
        _get_reconfigure_handle,
        _new_window,
        _rank_not_in_group,
        _reconfigure,
        _reduce_scatter_base,
        _supports_reconfigure,
        _supports_window,
        _time_estimator,
        get_node_local_rank,
    )
    from .remote_device import _remote_device
    from .rendezvous import (
        _create_store_from_options,
        register_rendezvous_handler,
        rendezvous,
    )

    set_debug_level_from_env()

else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    # We cannot define stubs directly because they confuse pyre

    class _Stub:
        # [CLASS DOC] HELPER: Empty placeholder type standing in for ProcessGroup and
        # GroupName when distributed support is not built.
        pass

    sys.modules["torch.distributed"].GroupName = _Stub  # type: ignore[attr-defined]
    sys.modules["torch.distributed"].ProcessGroup = _Stub  # type: ignore[attr-defined]
