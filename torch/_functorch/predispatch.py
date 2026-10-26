# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Pre-dispatch wrappers around functorch/forward-AD C++ primitives (batch dim
# add/remove, vmap and jvp nesting, dual levels, unwrap for grad) plus lazy loading of vmap
# decompositions. Each wrapper checks for an active pre-dispatch export mode; if present the
# call is routed through the torch function protocol so it is recorded in the exported graph,
# otherwise the underlying C++ implementation runs directly.
# Import-time side effects: none beyond imports; module globals track decomposition loading.
"""
This module contains pre-dispatch wrappers for functorch operations
that enable proper tracing in PT2 non-strict export/compile fx graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._C import (
    _enter_dual_level as _enter_dual_level_impl,
    _exit_dual_level as _exit_dual_level_impl,
)
from torch._C._functorch import (
    _add_batch_dim as _add_batch_dim_impl,
    _jvp_decrement_nesting as _jvp_decrement_nesting_impl,
    _jvp_increment_nesting as _jvp_increment_nesting_impl,
    _remove_batch_dim as _remove_batch_dim_impl,
    _unwrap_for_grad as _unwrap_for_grad_impl,
    _vmap_decrement_nesting as _vmap_decrement_nesting_impl,
    _vmap_increment_nesting as _vmap_increment_nesting_impl,
)
from torch._VF import (  # type: ignore[attr-defined]
    _make_dual as _make_dual_impl,
    _unpack_dual as _unpack_dual_impl,
)


if TYPE_CHECKING:
    import threading


def _add_batch_dim(self: torch.Tensor, batch_dim: int, level: int) -> torch.Tensor:
    # [FUNCTION DOC] Wraps a tensor as batched at the given vmap level, with batch_dim being
    # the dimension to treat as the batch dimension (negative values count from the end).
    # Returns the batched tensor. Under a pre-dispatch export mode it dispatches through
    # handle_torch_function so the call is traced; otherwise calls the C++ implementation.
    """
    Thin wrapper around torch._C._add_batch_dim that is used to proxy in
    PT2 export/compile fx graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()
    batch_dim = self.ndim + batch_dim if batch_dim < 0 else batch_dim

    if mode:
        return torch.overrides.handle_torch_function(
            _add_batch_dim, (self,), self, batch_dim, level
        )

    res = _add_batch_dim_impl(self, batch_dim, level)
    return res


def _remove_batch_dim(
    self: torch.Tensor, level: int, batch_size: int, out_dim: int
) -> torch.Tensor:
    # [FUNCTION DOC] Removes the batch dimension of the given vmap level from a tensor, placing
    # it at out_dim with the given batch_size. Returns the unbatched tensor. Traced through
    # handle_torch_function under a pre-dispatch export mode, else calls the C++ implementation.
    """
    Thin wrapper around torch._C._remove_batch_dim that is used to proxy in
    PT2 export/compile fx graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _remove_batch_dim, (self,), self, level, batch_size, out_dim
        )

    res = _remove_batch_dim_impl(self, level, batch_size, out_dim)
    return res


def _vmap_increment_nesting(batch_size: int, randomness: str) -> int:
    # [FUNCTION DOC] Enters a new vmap nesting level for the given batch_size and randomness
    # mode; returns the new level as an int. Side effect: pushes a functorch dynamic layer.
    # Traced under a pre-dispatch export mode, else calls the C++ implementation.
    """
    Thin wrapper around torch._C._vmap_increment_nesting that is used
    to proxy in export/compile graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _vmap_increment_nesting, (batch_size,), batch_size, randomness
        )
    res = _vmap_increment_nesting_impl(batch_size, randomness)
    return res


def _vmap_decrement_nesting() -> int:
    # [FUNCTION DOC] Exits the innermost vmap nesting level and returns the level that was
    # popped. Side effect: pops a functorch dynamic layer. Traced under a pre-dispatch export
    # mode, else calls the C++ implementation.
    # KNOWN ISSUE: the docstring says it wraps the increment function; it wraps the decrement one.
    """
    Thin wrapper around torch._C._vmap_increment_nesting that is used
    to proxy in export/compile graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _vmap_decrement_nesting,
            (),
        )
    return _vmap_decrement_nesting_impl()


# Global variables for lazy_load_decompositions
DECOMPOSITIONS_LOADED: bool = False
DECOMPOSITIONS_LOCK: threading.Lock | None = None
VMAP_DECOMPOSITIONS_LIB: torch.library.Library | None = None


def lazy_load_decompositions() -> None:
    # [FUNCTION DOC] Registers Python decompositions (mse/smooth_l1/huber loss backward, nll_loss
    # forward/backward in 1d and 2d, addr) for the FuncTorchBatched dispatch key, at most once.
    # Uses a lazily created lock with double-checked flag to be thread safe. Skipped (flag set,
    # nothing registered) unless PYTORCH_JIT is "1" and asserts are enabled. Under a pre-dispatch
    # export mode it only dispatches through handle_torch_function. Mutates module globals
    # DECOMPOSITIONS_LOADED, DECOMPOSITIONS_LOCK, VMAP_DECOMPOSITIONS_LIB. Returns None.
    # Raises RuntimeError if a decomposition is missing from the decomposition table.
    """
    Lazy loading of vmap decompositions with pre-dispatch support.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(lazy_load_decompositions, ())

    global DECOMPOSITIONS_LOADED, DECOMPOSITIONS_LOCK, VMAP_DECOMPOSITIONS_LIB

    if DECOMPOSITIONS_LOADED:
        return

    # Initialize lock if needed
    if DECOMPOSITIONS_LOCK is None:
        import threading

        DECOMPOSITIONS_LOCK = threading.Lock()

    with DECOMPOSITIONS_LOCK:
        if DECOMPOSITIONS_LOADED:
            return

        import os

        if not (os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__):
            DECOMPOSITIONS_LOADED = True
            return

        # use an alternate way to register an operator into the decomposition table
        # _register_jit_decomposition doesn't work for some operators, e.g. addr,
        #  because the Tensor types generated cannot be unioned by torchscript
        # decomp should be type OpOverload
        VMAP_DECOMPOSITIONS_LIB = torch.library.Library(
            "aten", "IMPL", "FuncTorchBatched"
        )

        from torch._decomp import decomposition_table

        def _register_python_decomposition_vmap(decomp: torch._ops.OpOverload) -> None:
            # [FUNCTION DOC] HELPER: Registers the table decomposition of an operator overload as
            # its FuncTorchBatched implementation in the vmap library. Raises AssertionError if the
            # library is unset and RuntimeError if the overload has no decomposition.
            if VMAP_DECOMPOSITIONS_LIB is None:
                raise AssertionError("VMAP_DECOMPOSITIONS_LIB must not be None")
            if decomp in decomposition_table:
                VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
            else:
                raise RuntimeError(f"could not find decomposition for {decomp}")

        _register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
        _register_python_decomposition_vmap(
            torch.ops.aten.smooth_l1_loss_backward.default
        )
        _register_python_decomposition_vmap(torch.ops.aten.huber_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.addr.default)

        DECOMPOSITIONS_LOADED = True


def _make_dual(
    tensor: torch.Tensor, tangent: torch.Tensor, *, level: int = 0
) -> torch.Tensor:
    # [FUNCTION DOC] Creates a forward-AD dual tensor from a primal tensor and tangent at the
    # given dual level; returns the dual tensor. Traced under a pre-dispatch export mode,
    # else calls the C++ implementation.
    """
    Thin wrapper around torch._VF._make_dual that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _make_dual, (tensor, tangent), tensor, tangent, level=level
        )

    return _make_dual_impl(tensor, tangent, level=level)


def _unpack_dual(
    tensor: torch.Tensor, *, level: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Splits a dual tensor into its (primal, tangent) pair at the given dual
    # level. Traced under a pre-dispatch export mode, else calls the C++ implementation.
    """
    Thin wrapper around torch._VF._unpack_dual that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _unpack_dual, (tensor,), tensor, level=level
        )

    return _unpack_dual_impl(tensor, level=level)


def _jvp_increment_nesting() -> int:
    # [FUNCTION DOC] Enters a new jvp nesting level and returns the new level as an int.
    # Side effect: pushes a functorch dynamic layer. Traced under a pre-dispatch export mode.
    """
    Thin wrapper around torch._C._functorch._jvp_increment_nesting that is
    used to proxy in export/compile graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _jvp_increment_nesting,
            (),
        )
    return _jvp_increment_nesting_impl()


def _jvp_decrement_nesting() -> int:
    # [FUNCTION DOC] Exits the innermost jvp nesting level and returns the popped level.
    # Side effect: pops a functorch dynamic layer. Traced under a pre-dispatch export mode.
    """
    Thin wrapper around torch._C._functorch._jvp_decrement_nesting that is
    used to proxy in export/compile graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _jvp_decrement_nesting,
            (),
        )
    return _jvp_decrement_nesting_impl()


def _unwrap_for_grad(tensor: torch.Tensor, level: int) -> torch.Tensor:
    # [FUNCTION DOC] Unwraps a tensor from the grad wrapper of the given level, returning the
    # inner tensor. Traced under a pre-dispatch export mode, else calls the C++ implementation.
    """
    Thin wrapper around torch._C._functorch._unwrap_for_grad that is used
    to proxy in PT2 export/compile fx graph for functorch transforms (grad, vjp, jvp).
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _unwrap_for_grad, (tensor,), tensor, level
        )

    return _unwrap_for_grad_impl(tensor, level)


def _enter_dual_level() -> int:
    # [FUNCTION DOC] Enters a new forward-AD dual level and returns its integer level.
    # Side effect: changes global forward-AD state. Traced under a pre-dispatch export mode.
    """
    Thin wrapper around torch._C._enter_dual_level that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(_enter_dual_level, ())
    return _enter_dual_level_impl()


def _exit_dual_level(*, level: int) -> None:
    # [FUNCTION DOC] Exits the given keyword-only forward-AD dual level; returns None.
    # Side effect: changes global forward-AD state. Traced under a pre-dispatch export mode.
    """
    Thin wrapper around torch._C._exit_dual_level that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(_exit_dual_level, (), level=level)
    return _exit_dual_level_impl(level=level)
