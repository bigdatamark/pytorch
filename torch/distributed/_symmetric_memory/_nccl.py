# [MODULE DOC] Helpers to publish an externally-owned NCCL communicator into the
# symmetric memory registry. Exports NcclCommRegistration (a handle that
# unregisters on release) and register_external_nccl_comm (the registration
# entry point). Import has no side effects; C extension functions are imported
# lazily at call time.
from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Self

import torch


if TYPE_CHECKING:
    from torch.types import _device


__all__ = ["NcclCommRegistration", "register_external_nccl_comm"]


class NcclCommRegistration:
    # [CLASS DOC] Handle for one registered external NCCL comm. State: group name,
    # device, optional strong reference to the source comm, and an active flag.
    # Usable as a context manager; unregisters on exit, on explicit unregister,
    # or on garbage collection. Does not own the comm itself.
    r"""
    Handle returned by :func:`register_external_nccl_comm`.

    Keeps an externally-owned NCCL communicator published in PyTorch's
    symmetric memory registry for as long as this object is alive. Dropping it
    (via ``del``, exiting its ``with`` block, or calling :meth:`unregister`)
    removes the entry so a successor producer can register cleanly under the
    same ``group_name``.

    The comm's lifetime is *not* owned by this object; the registration only
    publishes a borrowed pointer. Optionally a strong reference to the source
    comm object is held so a stray ``del comm`` cannot dangle the registered
    pointer for the lifetime of the registration.
    """

    def __init__(
        self,
        group_name: str,
        device: torch.device,
        comm: object | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the group name, device and optional comm reference
        # and marks the registration active. Performs no registration itself.
        self._group_name = group_name
        self._device = device
        self._comm = comm
        self._active = True

    def unregister(self) -> None:
        # [FUNCTION DOC] Removes the comm entry for this group and device from the
        # registry via the lazily imported C function, then marks the handle
        # inactive and drops the comm reference. No-op if already inactive.
        # Raises ImportError if the build lacks symmetric-memory NCCL support.
        """Remove the registration. Idempotent."""
        if not self._active:
            return
        # Imported lazily: only present in NCCL builds with symmetric-memory
        # device support.
        from torch._C._distributed_c10d import _unregister_external_nccl_comm

        _unregister_external_nccl_comm(self._group_name, self._device)
        self._active = False
        self._comm = None

    def __enter__(self) -> Self:
        # [FUNCTION DOC] Context manager entry; returns this registration.
        return self

    def __exit__(self, *exc: object) -> None:
        # [FUNCTION DOC] Context manager exit; unregisters. Exceptions propagate.
        self.unregister()

    def __del__(self) -> None:
        # [FUNCTION DOC] Best-effort unregister on finalization; swallows any
        # Exception so teardown never raises.
        try:
            self.unregister()
        except Exception:
            # Best-effort cleanup; never raise from __del__ (e.g. during
            # interpreter teardown the C extension may already be gone).
            pass


def register_external_nccl_comm(
    group_name: str,
    comm_ptr: int,
    device: _device,
    comm: object | None = None,
) -> NcclCommRegistration:
    # [FUNCTION DOC] Converts device to a torch device, registers the raw NCCL comm
    # pointer under group_name via the lazily imported C function (side effect:
    # mutates the global registry), and returns a NcclCommRegistration holding the
    # optional comm reference. Raises ImportError without NCCL symmetric-memory support.
    r"""
    Publish an externally-owned ``ncclComm_t`` into PyTorch's symmetric memory
    registry under ``group_name``.

    This lets producers that are not a ``ProcessGroupNCCL`` (e.g. torchcomms
    backends) supply the host NCCL communicator that
    :func:`rendezvous` needs, without symmetric memory having to
    ``dynamic_cast`` to a specific process-group implementation. After this
    call, ``rendezvous(tensor, group=group_name)`` on a tensor in a process
    group registered under ``group_name`` will find this comm.

    Args:
        group_name (str): the process-group name to register the comm under.
        comm_ptr (int): the host ``ncclComm_t`` as an opaque integer pointer
            (e.g. from ``torchcomms`` ``get_nccl_comm_ptr()``).
        device (`torch.device` or str): the CUDA device the comm belongs to.
        comm (object, optional): the source comm object. When provided, a
            strong reference is held by the returned registration so the comm
            cannot be garbage-collected while still registered.

    Returns:
        NcclCommRegistration: keep this alive for as long as symmetric memory
        should use this comm; drop it (or call ``unregister()``) to remove the
        registration.

    .. note::
        This is only available in NCCL builds with symmetric-memory device
        support; otherwise it raises :class:`ImportError`.
    """
    from torch._C._distributed_c10d import (
        _register_external_nccl_comm as _register_external_nccl_comm_impl,
    )

    device = torch.device(device)
    _register_external_nccl_comm_impl(group_name, comm_ptr, device)
    return NcclCommRegistration(group_name, device, comm)
