# [MODULE DOC] Thread-local tracking of whether freezing is active, plus helpers to mark and query
# frozen parameters on tensors and frozen-params state on graph modules.
import contextlib
import threading
from collections.abc import Generator
from typing import Any

import torch


_TLS = threading.local()


def _freezing_active() -> bool:
    # [FUNCTION DOC] HELPER: Returns whether freezing is active in the current thread (default False).
    return getattr(_TLS, "freezing_active", False)


@contextlib.contextmanager
def enter_freezing() -> Generator[Any, None, None]:
    # [FUNCTION DOC] Context manager: marks freezing active for the current thread inside the block and
    # restores the previous value on exit (even on error). Side effect: thread-local state.
    """
    Context manager to designate when freezing is active.
    """
    prev = _freezing_active()
    _TLS.freezing_active = True
    try:
        yield
    finally:
        _TLS.freezing_active = prev


def record_has_frozen_params(gm: torch.fx.GraphModule) -> None:
    # [FUNCTION DOC] Sets a flag on the graph module indicating it has frozen params.
    """
    Mark the gm as having frozen params.
    """
    gm._has_frozen_params = True  # type: ignore[assignment]


def has_frozen_params(gm: torch.fx.GraphModule) -> bool:
    # [FUNCTION DOC] Returns whether the graph module has been flagged as having frozen params.
    """
    Return True if the gm has frozen parameters.
    """
    return getattr(gm, "_has_frozen_params", False)


def maybe_set_is_frozen_param(t: torch.Tensor) -> None:
    # [FUNCTION DOC] If freezing is active in this thread, flags the tensor as a frozen parameter.
    """
    Mark the provided tensor as a frozen param if freezing is active.
    """
    if _freezing_active():
        t._is_frozen_param = True  # type: ignore[attr-defined]


def is_frozen_param(t: torch.Tensor) -> bool:
    # [FUNCTION DOC] Returns whether the tensor was flagged as a frozen parameter.
    """
    Return True if the tensor is a frozen param.
    """
    return getattr(t, "_is_frozen_param", False)
