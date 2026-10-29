# [MODULE DOC] Thread-local dynamic scoping of a few global code generation toggles.
# Provides accessors that require initialization and a parametrize context manager.
# No import-time side effects beyond creating the module-level thread-local instance.
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.
#
# WARNING WARNING: LOOKING TO EDIT THIS FILE?  Think carefully about
# why you need to add a toggle to the global behavior of code
# generation.  The parameters here should really only be used
# for "temporary" situations, where we need to temporarily change
# the codegen in some cases because we cannot conveniently update
# all call sites, and are slated to be eliminated once all call
# sites are eliminated.  If you don't have a plan for how to get there,
# DON'T add a new entry here.


class Locals(threading.local):
    # [CLASS DOC] Thread-local holder for the two codegen toggles; each is None until set by
    # parametrize. None means uninitialized and makes the accessors raise an error.
    use_const_ref_for_mutable_tensors: bool | None = None
    use_ilistref_for_tensor_lists: bool | None = None


_locals = Locals()


def use_const_ref_for_mutable_tensors() -> bool:
    # [FUNCTION DOC] Return the current thread's const-ref toggle for mutable tensors.
    # Raises an AssertionError if parametrize has not set it yet.
    if _locals.use_const_ref_for_mutable_tensors is None:
        raise AssertionError(
            "need to initialize local.use_const_ref_for_mutable_tensors with "
            "local.parametrize"
        )
    return _locals.use_const_ref_for_mutable_tensors


def use_ilistref_for_tensor_lists() -> bool:
    # [FUNCTION DOC] Return the current thread's ilistref toggle for tensor lists.
    # Raises an AssertionError if parametrize has not set it yet.
    if _locals.use_ilistref_for_tensor_lists is None:
        raise AssertionError(
            "need to initialize local.use_ilistref_for_tensor_lists with local.parametrize"
        )
    return _locals.use_ilistref_for_tensor_lists


@contextmanager
def parametrize(
    *, use_const_ref_for_mutable_tensors: bool, use_ilistref_for_tensor_lists: bool
) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that sets both codegen toggles for the current thread.
    # Saves the previous values, yields nothing, and restores them on exit even on error.
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_use_ilistref_for_tensor_lists = _locals.use_ilistref_for_tensor_lists
    try:
        _locals.use_const_ref_for_mutable_tensors = use_const_ref_for_mutable_tensors
        _locals.use_ilistref_for_tensor_lists = use_ilistref_for_tensor_lists
        yield
    finally:
        _locals.use_const_ref_for_mutable_tensors = (
            old_use_const_ref_for_mutable_tensors
        )
        _locals.use_ilistref_for_tensor_lists = old_use_ilistref_for_tensor_lists
