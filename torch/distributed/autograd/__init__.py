# [PACKAGE DOC]
# Python front end for distributed autograd. When the C++ extension is built with
# distributed support, re-exports the native context helpers and backward, get_gradients,
# DistAutogradContext. Defines the context manager class (creates and releases a distributed
# autograd context id) and is_available. Import-time side effect: initializes the native
# module and raises RuntimeError if that initialization fails. Public __all__: context, is_available.
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from types import TracebackType


def is_available() -> bool:
    # [FUNCTION DOC] Returns True if the torch C extension was built with distributed autograd support.
    return hasattr(torch._C, "_dist_autograd_init")


if is_available() and not torch._C._dist_autograd_init():
    raise RuntimeError("Failed to initialize torch.distributed.autograd")

if is_available():
    from torch._C._distributed_autograd import (
        _current_context,
        _get_debug_info,
        _get_max_id,
        _init,
        _is_valid_context,
        _new_context,
        _release_context,
        _retrieve_context,
        backward,
        DistAutogradContext,
        get_gradients,
    )

__all__ = ["context", "is_available"]


class context:
    # [CLASS DOC] Context manager wrapping a distributed autograd pass. Entering creates a
    # new native context (kept in autograd_context) and yields its integer id; exiting
    # releases that context. Used with rpc calls and the distributed backward function.
    """
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = torch.rand((3, 3), requires_grad=True)
        >>>     t2 = torch.rand((3, 3), requires_grad=True)
        >>>     loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
        >>>     dist_autograd.backward(context_id, [loss])
    """

    def __enter__(self) -> int:
        # [FUNCTION DOC] Creates a new distributed autograd context, stores it, returns its integer id.
        self.autograd_context = _new_context()
        return self.autograd_context._context_id()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Releases the context created on entry; exception info is ignored (not suppressed).
        _release_context(self.autograd_context._context_id())
