# mypy: allow-untyped-defs
# [MODULE DOC] Implements unsharding of FSDP parameters for summon_full_params-style use:
# context managers that unshard (and optionally offload, write back, include grads) the
# flat parameters of one or many FSDP states, plus helpers to register/deregister the flat
# parameter and original parameters on a module. Defines FLAT_PARAM, the attribute name.
import contextlib
import warnings
from collections.abc import Generator
from typing import cast

import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state,
    _has_fsdp_params,
    _module_handle,
    HandleTrainingState,
    TrainingState,
)
from torch.distributed.fsdp._runtime_utils import (
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
    _reshard,
    _reshard_grads,
    _unshard,
    _unshard_grads,
)
from torch.distributed.utils import _p_assert

from ._flat_param import FlatParamHandle


FLAT_PARAM = "_flat_param"


@torch.no_grad()
def _writeback_to_local_shard(
    handle: FlatParamHandle,
    writeback_grad: bool,
):
    # [FUNCTION DOC] HELPER: Under no-grad, copy this rank's shard of the unsharded flat parameter
    # back into the stored local shard (and, if writeback_grad and a sharded grad exists, the
    # grad shard). Sharded strategies use the unpadded shard; NO_SHARD uses the tensor as is.
    # Raises AssertionError if a sharded grad exists but flat_param.grad is None.
    """
    For the handle, writes back this rank's shard of the unsharded
    flattened parameter to the sharded flattened parameter. If
    ``writeback_grad=True``, then writes back to the sharded gradient as
    well.

    Precondition: The handle's ``FlatParameter`` 's data points to the
    padded unsharded flattened parameter.
    """

    def _get_shard(flat_param_or_grad: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Return this rank's unpadded shard of the tensor for sharded
        # strategies, otherwise the tensor itself.
        if handle.uses_sharded_strategy:
            # For sharded strategies, get the *unpadded* shard instead of
            # the *padded* shard to persist user changes to the padding
            # (though FSDP does not explicitly support this)
            shard, _ = FlatParamHandle._get_unpadded_shard(
                flat_param_or_grad,
                handle.rank,
                handle.world_size,
            )
            return shard
        # For `NO_SHARD`, the `flat_param` or its gradient may be modified,
        # so we write it back directly
        return flat_param_or_grad

    param_shard = _get_shard(handle.flat_param)
    handle.flat_param._local_shard[: param_shard.numel()].copy_(param_shard)  # type: ignore[attr-defined]
    if writeback_grad:
        existing_grad = handle.sharded_grad
        if existing_grad is not None:
            if handle.flat_param.grad is None:
                raise AssertionError("Expected handle.flat_param.grad to not be None")
            grad_shard = _get_shard(handle.flat_param.grad)
            existing_grad[: grad_shard.numel()].copy_(grad_shard)


def _deregister_flat_param(state: _FSDPState, module: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: If the module has FSDP params, remove the flat parameter entry from
    # the wrapped module's parameter mapping (no error if absent). Mutates the module.
    """
    De-registers the flattened parameter from the wrapped module, hiding it
    from ``nn.Module`` methods.

    We do not use ``del`` because we want ``FLAT_PARAM`` to always be an
    attribute but dynamically change whether it is visible to ``nn.Module``
    methods.
    """
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        cast(nn.Module, module.module)._parameters.pop(FLAT_PARAM, None)


def _register_flat_param(state: _FSDPState, module: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: If the module has FSDP params, put the handle's flat parameter into
    # the wrapped module's parameter mapping under FLAT_PARAM. Mutates the module.
    """
    Registers the flattened parameter to the wrapped module, making it
    visible to ``nn.Module`` methods.

    We do not use :meth:`nn.Module.register_parameter` because we want
    ``FLAT_PARAM`` to always be an attribute but dynamically change whether
    it is visible to ``nn.Module`` methods.
    """
    handle = _module_handle(state, module)
    if _has_fsdp_params(state, module):
        # TODO: figure out the case for the composable APIs.
        cast(nn.Module, module.module)._parameters[FLAT_PARAM] = handle.flat_param


@contextlib.contextmanager
def _unflatten_as_params(state: _FSDPState, module: nn.Module) -> Generator:
    # [FUNCTION DOC] HELPER: Context manager (flat parameter must be unsharded). With no handle it
    # just yields. Otherwise deregisters the flat parameter, enters the handle's
    # unflatten-as-params context, and on exit re-registers the flat parameter unless
    # original parameters are in use.
    """
    Assumes that the flattened parameter is unsharded. When in the context,
    de-registers the flattened parameter and unflattens the original
    parameters as ``nn.Parameter`` views into the flattened parameter.
    After the context, re-registers the flattened parameter and restores
    the original parameters as ``Tensor`` views into the flattened
    parameter.
    """
    handle = _module_handle(state, module)
    if not handle:
        yield
    else:
        _deregister_flat_param(state, module)
        try:
            with handle.unflatten_as_params():
                yield
        finally:
            if not handle._use_orig_params:
                _register_flat_param(state, module)


def _validate_unshard_params_args(
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
) -> None:
    # [FUNCTION DOC] HELPER: Validate argument combinations. Raises NotImplementedError for
    # with_grads with offload_to_cpu or without original params, offload_to_cpu with
    # NO_SHARD, and writeback with rank0_only. Warns if offload_to_cpu without rank0_only.
    if with_grads and (offload_to_cpu or not state._use_orig_params):
        raise NotImplementedError(
            f"with_grads={with_grads}, "
            f"use_orig_params={state._use_orig_params}, "
            f"offload_to_cpu={offload_to_cpu} "
            f"is not supported yet"
        )
    if offload_to_cpu and state._handle and (not state._handle.uses_sharded_strategy):
        raise NotImplementedError(
            "offload_to_cpu=True and NO_SHARD is not supported yet"
        )
    if writeback and rank0_only:
        # TODO: Rank 0 can broadcast the `FlatParameter` to allow all ranks to
        # persist the changes.
        raise NotImplementedError(
            "writeback=True and rank0_only=True is not supported yet"
        )
    if offload_to_cpu and not rank0_only:
        warnings.warn(
            "offload_to_cpu=True and rank0_only=False may result in the"
            "unsharded parameters being redundantly copied to CPU memory for "
            "GPUs sharing the same CPU memory, which risks CPU OOM. We "
            "recommend using offload_to_cpu=True with rank0_only=True.",
            stacklevel=2,
        )


@contextlib.contextmanager
def _unshard_fsdp_state_params(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    # [FUNCTION DOC] HELPER: Context manager unsharding the parameters of one FSDP state for the
    # given module. Validates args, synchronizes the device, and yields untouched if there
    # is no handle or it is already in summon state. Otherwise marks the handle as summoning,
    # unshards in the current stream (plus grads if requested); on rank != 0 with rank0_only
    # it reshards immediately and yields. Else optionally moves to CPU and unflattens, and
    # on exit writes back (if writeback), reshards, and resets the handle to IDLE.
    # Raises AssertionError if the handle is not IDLE at entry.
    """
    This unshards the parameters for a single FSDP state ``state`` that
    corresponds to ``module``.
    """
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    state._device_handle.synchronize()
    # If handles are shared by other module(s), the handle may be already unsharded.
    maybe_handle = _module_handle(state, module)
    handle = None
    if (
        maybe_handle
        and maybe_handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS
    ):
        handle = maybe_handle
    if not handle:
        yield
        return

    if handle._training_state != HandleTrainingState.IDLE:
        raise AssertionError(
            f"Expects the handle training to be IDLE but got {handle._training_state}"
        )

    handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS

    _reset_flat_param_grad_info_if_needed(handle)
    free_unsharded_flat_param = handle.needs_unshard()
    # No need to call `wait_stream()` since we unshard in the computation
    # stream directly
    computation_stream = state._device_handle.current_stream()
    _unshard(state, handle, computation_stream, computation_stream)
    if with_grads:
        _unshard_grads(handle)

    if rank0_only and state.rank != 0:
        # Free the unsharded flattened parameter early
        _reshard(state, handle, free_unsharded_flat_param)
        if with_grads:
            _reshard_grads(handle)
        try:
            yield
        finally:
            handle._training_state = HandleTrainingState.IDLE
    else:
        # Unflatten the unsharded flattened parameters
        with contextlib.ExitStack() as stack:
            # Invariant: rank == 0 or !rank0_only
            if offload_to_cpu and handle.uses_sharded_strategy:
                stack.enter_context(handle.to_cpu())
                # NOTE: Since PyTorch enforces that a parameter and its
                # gradients need to match metadata (e.g. device), we must
                # move gradients to CPU *after* we move parameters.
            # NOTE: This assumes 1 `FlatParameter`
            if not state._use_orig_params:
                stack.enter_context(_unflatten_as_params(state, module))
            try:
                yield
            finally:
                stack.close()
                if writeback:
                    _writeback_to_local_shard(handle, with_grads)
                _reshard(state, handle, free_unsharded_flat_param)
                if with_grads:
                    _reshard_grads(handle)
                handle._training_state = HandleTrainingState.IDLE


@contextlib.contextmanager
def _unshard_params_for_summon(
    module: nn.Module,
    state: _FSDPState,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    # [FUNCTION DOC] HELPER: Context manager for one state: validates args, runs lazy init, and
    # raises AssertionError if the state is in forward/backward or already summoning. Enters
    # the per-state unshard context and sets training state to SUMMON_FULL_PARAMS for the
    # duration, resetting it to IDLE on exit.
    _validate_unshard_params_args(
        state, writeback, rank0_only, offload_to_cpu, with_grads
    )
    _lazy_init(state, module)
    if state.training_state == TrainingState.FORWARD_BACKWARD:
        raise AssertionError(
            "Cannot manually unshard parameters during forward/backward"
        )
    elif state.training_state == TrainingState.SUMMON_FULL_PARAMS:
        raise AssertionError(
            "Cannot manually unshard parameters when already unsharding parameters"
        )
    with _unshard_fsdp_state_params(
        module=module,
        state=state,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    ):
        try:
            state.training_state = TrainingState.SUMMON_FULL_PARAMS
            yield
        finally:
            state.training_state = TrainingState.IDLE


@contextlib.contextmanager
def _unshard_params(
    module: nn.Module,
    recurse: bool,
    writeback: bool,
    rank0_only: bool,
    offload_to_cpu: bool,
    with_grads: bool,
):
    # [FUNCTION DOC] HELPER: Context manager unsharding all FSDP-managed parameters under module
    # (only module's own state when recurse is False; yields untouched if it has none).
    # Enters a per-state summon context for each (state, module) pair in one exit stack.
    """
    This unshards FSDP-managed parameters for all modules with FSDP applied in
    the module tree rooted at ``module``.
    """
    if not recurse:
        optional_state = _get_module_fsdp_state(module)
        if optional_state is None:
            with contextlib.nullcontext():
                yield
            return
        states_and_modules = ([optional_state], [module])
    else:
        states_and_modules = traversal_utils._get_fsdp_states_with_modules(module)
    with contextlib.ExitStack() as stack:
        for state, module in zip(*states_and_modules):
            stack.enter_context(
                _unshard_params_for_summon(
                    module=module,
                    state=state,
                    writeback=writeback,
                    rank0_only=rank0_only,
                    offload_to_cpu=offload_to_cpu,
                    with_grads=with_grads,
                )
            )
        yield


def _deregister_orig_params(state: _FSDPState, module: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: No-op without a handle. Otherwise asserts (via _p_assert) that the
    # state and handle agree on using original params, has the handle deregister the original
    # parameters, and registers the flat parameter on the module.
    """
    Deregisters the original parameters; registers the ``FlatParameter``.
    """
    handle = _module_handle(state, module)
    if not handle:
        return
    _p_assert(
        handle._use_orig_params,
        f"Inconsistent `_use_orig_params` -- FSDP: {state._use_orig_params} "
        f"handle: {handle._use_orig_params}",
    )
    handle._deregister_orig_params()
    _register_flat_param(state, module)


def _register_orig_params(state: _FSDPState, module: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: No-op without a handle. Otherwise deregisters the flat parameter
    # and restores original parameters as views: sharded views (and grad views) if the flat
    # parameter is sharded, else unsharded views as parameters.
    """
    Deregisters the ``FlatParameter``; registers the original parameters.
    """
    handle = _module_handle(state, module)
    if not handle:
        return
    _deregister_flat_param(state, module)
    if handle.is_sharded(handle.flat_param):
        handle._use_sharded_views()
        handle._use_sharded_grad_views()
    else:
        handle._use_unsharded_views(as_params=True)
