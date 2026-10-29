# mypy: allow-untyped-defs
# [MODULE DOC] Runtime logic of FSDP: lazy initialization of the root state and shared
# streams, unshard/reshard of flat-parameter handles, pre/post forward and backward hooks,
# gradient reduce-scatter/all-reduce, prefetching, mixed-precision input/buffer casting, and
# helpers to find FSDP roots and registered hooks. Shared by the wrapper FullyShardedDataParallel
# and composable fully_shard paths. Import creates a module logger and the constant
# HOMOGENEOUS_ATTR_NAMES; no other side effects.
import functools
import logging
from collections.abc import Callable
from enum import auto, Enum
from typing import Any, no_type_check

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS
from torch.distributed.fsdp._common_utils import (
    _assert_in_training_states,
    _FSDPState,
    _get_module_fsdp_state,
    _is_composable,
    _log_post_backward_hook,
    _no_dispatch_record_stream,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp._flat_param import (
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
    RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.utils import (
    _apply_to_tensors,
    _cast_forward_inputs,
    _p_assert,
    _to_kwargs,
)
from torch.utils import _pytree as pytree


logger = logging.getLogger(__name__)

# Do not include "process_group" to enable hybrid shard and MoE cases
HOMOGENEOUS_ATTR_NAMES = (
    "_use_orig_params",
    "limit_all_gathers",
    "_use_full_prec_in_eval",
)


class _PrefetchMode(Enum):
    # [CLASS DOC] HELPER: Enum of prefetch directions, BACKWARD and FORWARD, selecting which
    # execution order is used to find the next handle to prefetch.
    BACKWARD = auto()
    FORWARD = auto()


def _get_fsdp_root_states_with_modules(
    module: nn.Module,
) -> tuple[list[_FSDPState], list[nn.Module]]:
    # [FUNCTION DOC] Walk module.modules() (assumed top-down) and collect unique FSDP states that
    # are roots, plus their owning modules, as two parallel lists. Forces lazy init on each
    # candidate to decide root-ness, which may raise errors from lazy init.
    """
    Returns a tuple containing:
    1. A list of the root ``_FSDPState`` instances in the module tree rooted at
    ``module`` without any duplicates and following the ``module.modules()``
    traversal order (which is assumed to be depth-first).
    2. A corresponding list of the root modules owning the states in the first
    list.

    This is similar to :func:`_get_fsdp_states_with_modules` except that we
    must call :func:`_is_fsdp_root` to force a lazy initialization to determine
    the FSDP root in case lazy initialization has not yet happened.
    """
    fsdp_root_states: list[_FSDPState] = []
    fsdp_root_modules: list[nn.Module] = []
    visited_fsdp_states: set[_FSDPState] = set()
    # NOTE: This function assumes that `module.modules()` proceeds top-down.
    for submodule in module.modules():
        optional_state = _get_module_fsdp_state(submodule)
        if (
            optional_state is not None
            and optional_state not in visited_fsdp_states
            and _is_fsdp_root(optional_state, submodule)
        ):
            visited_fsdp_states.add(optional_state)
            fsdp_root_states.append(optional_state)
            fsdp_root_modules.append(submodule)
    return fsdp_root_states, fsdp_root_modules


def _get_fsdp_root_states(module: nn.Module) -> list[_FSDPState]:
    # [FUNCTION DOC] HELPER: Return only the states from _get_fsdp_root_states_with_modules.
    """See :func:`_get_fsdp_root_states_with_modules`."""
    fsdp_root_states, _ = _get_fsdp_root_states_with_modules(module)
    return fsdp_root_states


def _is_fsdp_root(state: _FSDPState, module: nn.Module) -> bool:
    # [FUNCTION DOC] HELPER: Force lazy init and return the state's is-root flag. Raises
    # AssertionError if the flag is still unset afterwards.
    """
    Returns if ``state`` corresponds to that of an FSDP root.

    For the wrapper code path, ``state`` and ``module`` should be the same. For
    the non-wrapper code path, ``state`` should be ``module`` 's state.
    """
    # Force a lazy initialization to determine the FSDP root
    _lazy_init(state, module)
    if state._is_root is None:
        raise AssertionError("Expected _is_root to be set after lazy init")
    return state._is_root


@no_type_check
def _lazy_init(
    state: _FSDPState,
    root_module: nn.Module,
) -> _FSDPState:
    # [FUNCTION DOC] Lazy initialization, run once at the first use. No-op if the state's root flag
    # is already set. Raises RuntimeError if the device is unavailable. Otherwise marks the
    # state as root, asserts IDLE training state, checks flat parameter devices, collects all
    # FSDP states in the tree, creates streams, casts buffers to their compute dtype/device,
    # initializes execution order data, and shares state with non-root states. Returns the
    # state on first init.
    # KNOWN ISSUE: the early return when already initialized returns None, not the state,
    # contradicting the return annotation.
    """
    Performs initialization lazily, typically right before the first forward
    pass. The laziness is needed to ensure that the parameter device/dtype and
    the FSDP hierarchy have finalized. This method's actual logic only runs on
    the root FSDP instance, which performs initialization for all non-root FSDP
    instances to avoid partial initialization.

    For the non-composable code path, ``state`` and ``root_module`` should be
    the same, namely the FSDP instance itself.
    """
    if state._is_root is not None:
        return  # no-op: already lazily initialized
    if not state._device_handle.is_available():
        # Allow the FSDP constructor to run even without CUDA but check this
        # once we start real execution
        raise RuntimeError("FSDP does not support CPU only execution")
    # The following logic is only run on the root FSDP instance since it will
    # set `_is_root=False` for the non-root instances
    state._is_root = True
    _assert_in_training_states(state, [TrainingState.IDLE])
    _check_flat_params_on_expected_device(state, root_module)
    state._all_fsdp_states = traversal_utils._get_fsdp_states(root_module)
    _init_streams(state)
    buffers, buffer_dtypes = _get_buffers_and_dtypes_for_computation(state, root_module)
    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes, state.compute_device)
    state._exec_order_data.init(state, root_module, state.process_group)
    _share_state_and_init_handle_attrs(state, root_module)
    return state


def _check_flat_params_on_expected_device(state: _FSDPState, module: nn.Module):
    # [FUNCTION DOC] HELPER: For every handle in the module tree, raise RuntimeError if params are
    # not on the compute device (non-offloaded) or not on CPU (when offloading params).
    """
    Checks that all ``FlatParameter``s in ``module`` 's tree managed by
    ``state`` are on the expected device for *lazy initialization*.
    """
    cpu_device = torch.device("cpu")
    for handle in traversal_utils._get_fsdp_handles(module):
        if (
            not handle._offload_params
            and handle.flat_param.device != state.compute_device
        ):
            raise RuntimeError(
                "An FSDP-managed module unexpectedly has parameters on "
                f"{handle.flat_param.device}. Make sure to move the module to "
                f"{state.compute_device} before training."
            )
        elif handle._offload_params and handle.flat_param.device != cpu_device:
            raise RuntimeError(
                "An FSDP-managed module with parameter CPU offloading enabled "
                f"has parameters on {handle.flat_param.device}. Make sure to "
                f"not move the module from CPU when offloading parameters."
            )


@no_type_check
def _share_state_and_init_handle_attrs(
    root_state: _FSDPState,
    root_module: nn.Module,
) -> None:
    # [FUNCTION DOC] HELPER: From the root, initialize its handle attributes, share the list of all
    # handles, and flag each handle's optimizer-in-backward status (RuntimeError if the
    # flat param itself has in-backward optimizers). For each non-root state: marks it non-root,
    # shares streams, exec order data, free event queue and extension compute stream, and
    # initializes its handle attributes. Raises ValueError if the homogeneous attributes
    # differ among states; asserts non-root states were not already roots.
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    handle = root_state._handle
    if handle:
        handle.init_flat_param_attributes()
    attr_name_to_values: dict[str, set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
    # Update _has_optim_in_backward for each handle.
    for handle in root_state._all_handles:
        flat_param = handle.flat_param
        if hasattr(flat_param, "_in_backward_optimizers"):
            raise RuntimeError(
                "FSDP optimizer in backward only supported with use_orig_params=True!"
            )
        handle._has_optim_in_backward = flat_param._params is not None and any(
            hasattr(param, "_in_backward_optimizers") for param in flat_param._params
        )
        if handle._has_optim_in_backward:
            torch._C._log_api_usage_once("fsdp.optimizer_in_backward")
    for fsdp_state in root_state._all_fsdp_states:
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(
                hasattr(fsdp_state, attr_name),
                f"FSDP state missing attribute {attr_name}",
            )
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        # Relax the assert for non-root FSDP instances in case the nested
        # initialized module is wrapped again in FSDP later (e.g. after
        # training to run inference)
        _p_assert(
            fsdp_state._is_root is None or not fsdp_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been "
            "set yet or should have been set to `False`",
        )
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
        fsdp_state._all_reduce_stream = root_state._all_reduce_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        if fsdp_state._fsdp_extension is not None:
            fsdp_state._fsdp_extension.compute_stream = root_state._default_stream
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()
    for attr_name, attr_values in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(
                f"Expects one homogeneous value for {attr_name} but got {attr_values}"
            )


@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    # [FUNCTION DOC] HELPER: On the root state, record the current stream as default compute
    # stream and create unshard, post-backward and pre-unshard streams (high priority if
    # limit_all_gathers with hybrid sharding), plus an all-reduce stream (new only for hybrid
    # sharding, else the default stream). Raises AssertionError if not root or no device.
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    if not state._is_root:
        raise AssertionError("Expected state to be root")
    if not state._device_handle.is_available():
        raise AssertionError("Expected device handle to be available")
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES
        for fsdp_state in state._all_fsdp_states
    )
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    if state._fsdp_extension is not None:
        # set the compute stream to the FSDP extension
        state._fsdp_extension.compute_stream = state._default_stream

    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream to run HSDP's all-reduce as async (if using HSDP)
    state._all_reduce_stream = (
        state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream
    )


@no_type_check
def _unshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    # [FUNCTION DOC] Unshard one handle (no-op if None): run pre-unshard on its stream, make the
    # unshard stream wait for it, optionally wait on the oldest free event (rate limiting),
    # then run unshard and post-unshard on the unshard stream. Leaves the flat parameter as
    # the padded unsharded data on the compute device.
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handle:
        return
    with state._device_handle.stream(pre_unshard_stream):
        ran_pre_unshard = handle.pre_unshard()
    if ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function(
                "FullyShardedDataParallel.rate_limiter"
            ):
                event.synchronize()
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()


@no_type_check
def _reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    free_unsharded_flat_param: bool,
):
    # [FUNCTION DOC] Reshard a handle, optionally freeing its unsharded flat parameter. With
    # limit_all_gathers and freeing (outside torch.compile), records and enqueues a free event.
    # Then runs post-reshard and clears the prefetched flag.
    """
    Reshards the handle. ``free_unsharded_flat_param`` indicates whether to
    free the handle's padded unsharded flat parameter.
    """
    handle.reshard(free_unsharded_flat_param)
    if state.limit_all_gathers and free_unsharded_flat_param:
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # We don't run an event queue for freeing under torch compile atm
            # But maybe we need to? TODO(voz): Look into this
            free_event = state._device_handle.Event()
            free_event.record()
            state._free_event_queue.enqueue(free_event)
    handle.post_reshard()
    # Flat parameter freed or not, we always have to "unshard" the parameter
    # upon next access to get its shape correct.
    handle._prefetched = False


def _unshard_grads(
    handle: FlatParamHandle | None,
) -> None:
    # [FUNCTION DOC] HELPER: Unshard the handle's gradient if a handle is given.
    if handle:
        handle.unshard_grad()


def _reshard_grads(
    handle: FlatParamHandle | None,
) -> None:
    # [FUNCTION DOC] HELPER: Reshard the handle's gradient if a handle is given.
    if handle:
        handle.reshard_grad()


@no_type_check
def _pre_forward(
    state: _FSDPState,
    handle: FlatParamHandle | None,
    unshard_fn: Callable,
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # [FUNCTION DOC] Pre-forward logic. Returns args and kwargs unchanged if the handle is in
    # BACKWARD_PRE (recomputed forward). Otherwise sets training states, records the forward
    # in exec order data, calls unshard_fn if given, registers post-backward hooks, reallocates
    # the pinned CPU grad buffer if params are offloaded and it was cleared, casts forward
    # inputs to the mixed-precision param dtype when enabled, registers the reshard-only
    # post-backward hook, and returns the possibly cast (args, kwargs).
    """
    Runs the pre-forward logic. This includes an opportunity to unshard
    currently sharded parameters such as those for the current forward and
    registering post-backward hooks for these current parameters. This function
    also converts forward ``args`` and ``kwargs`` to the given precision.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        unshard_fn (Optional[Callable]): A callable to unshard any currently
            sharded parameters or ``None`` to not do any unsharding.
        module (nn.Module): Module whose forward this method runs right before;
            expected by the hook signature.
        args (Tuple[Any, ...]): Module forward ``args``.
        kwargs (Dict[str, Any]): Module forward ``kwargs``.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._pre_forward"):
        # For `fully_shard` + `checkpoint`, skip pre-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            # For both checkpoint implementations, we do not need to re-cast
            # inputs here since they will be checkpointed in the low precision
            # either by AC or normally by autograd as long as the AC region is
            # nested within FSDP
            return args, kwargs
        state.training_state = TrainingState.FORWARD_BACKWARD
        state._exec_order_data.record_pre_forward(handle, module.training)
        if handle:
            handle._training_state = HandleTrainingState.FORWARD
        if unshard_fn is not None:
            unshard_fn(state, handle)
        # Register post-backward hooks to reshard the parameters and reduce-scatter
        # their gradients. They must be re-registered every forward pass in case
        # the `grad_fn` is mutated.
        _register_post_backward_hook(state, handle)
        # We have to reallocate the _cpu_grad if optimizer overlap
        # set the grad to None in the backward pass.
        if handle and handle._offload_params and handle.flat_param._cpu_grad is None:
            handle.flat_param._cpu_grad = torch.zeros_like(
                handle.flat_param._local_shard, device=torch.device("cpu")
            ).pin_memory()

        should_cast_forward_inputs = (
            state._handle and not state._handle._force_full_precision
        )

        if should_cast_forward_inputs and state.mixed_precision.cast_forward_inputs:
            # Recursively convert args and kwargs to specified precision.
            input_dtype: torch.dtype | None = state.mixed_precision.param_dtype
            args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)
        _register_post_backward_reshard_only_hook(state, handle, args, kwargs)
        return args, kwargs


@no_type_check
def _pre_forward_unshard(
    state: _FSDPState,
    handle: FlatParamHandle | None,
) -> None:
    # [FUNCTION DOC] Pre-forward unshard (no-op without handle): unshard unless prefetched, clear the
    # needs-pre-forward-unshard flag, make the current stream wait on the unshard event or
    # stream (skipped when tracing under torch.compile), then prefetch the next handle in
    # forward mode.
    """Unshards parameters in the pre-forward."""
    if not handle:
        return
    # If the handles have been prefetched, then there is no need to call
    # `_unshard()` again
    if not handle._prefetched:
        _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._needs_pre_forward_unshard = False
    # Don't wait during trace
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        current_stream = state._device_handle.current_stream()
        if state._unshard_event is not None:
            current_stream.wait_event(state._unshard_event)
            state._unshard_event = None
        else:
            current_stream.wait_stream(state._unshard_stream)
    with torch.profiler.record_function(
        "FullyShardedDataParallel._pre_forward_prefetch"
    ):
        _prefetch_handle(state, handle, _PrefetchMode.FORWARD)


@no_type_check
def _post_forward(
    state: _FSDPState,
    handle: FlatParamHandle | None,
    reshard_fn: Callable,
    module: nn.Module,
    input: Any,
    output: Any,
) -> Any:
    # [FUNCTION DOC] Post-forward logic. Returns output untouched if the handle is in BACKWARD_PRE.
    # Otherwise records the post-forward, calls reshard_fn if given, registers pre-backward
    # hooks on the output tensors, resets state and handle training states to IDLE, and
    # returns the (possibly rewrapped) output.
    """
    Runs the post-forward logic. This includes an opportunity to reshard
    currently unsharded parameters such as those used in the current forward
    and registering pre-backward hooks on the forward outputs.

    Args:
        handles (List[FlatParamHandle]): Handles giving the parameters used in
            the current forward.
        reshard_fn (Optional[Callable]): A callable to reshard any currently
            unsharded parameters (e.g. from the current forward) or ``None`` to
            not do any resharding.
        module (nn.Module): Module whose forward just ran, which should be a
            fully sharded module (see [Note: Fully Sharded Module]); expected
            by the hook signature.
        input (Any): Unused; expected by the hook signature.
        output (Any): Forward pass output; pre-backward hooks are registered on
            the tensors that require gradients in this output.

    Postcondition: Each ``FlatParameter`` 's data points to the sharded flat
    parameter.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._post_forward"):
        # For `fully_shard` + `checkpoint`, skip post-forward logic in the
        # recomputed forward
        if handle and handle._training_state == HandleTrainingState.BACKWARD_PRE:
            return output

        state._exec_order_data.record_post_forward(handle)
        if reshard_fn is not None:
            reshard_fn(state, handle)
        # Register pre-backward hooks to unshard the flat parameters for the
        # gradient computation (if needed)
        output = _register_pre_backward_hooks(state, module, output, handle)
        state.training_state = TrainingState.IDLE
        if handle:
            handle._training_state = HandleTrainingState.IDLE
        return output


@no_type_check
def _post_forward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> None:
    # [FUNCTION DOC] Post-forward reshard (no-op without handle). Frees the unsharded flat param
    # only for non-root states whose strategy reshards after forward (the root keeps its
    # params for the imminent backward).
    """Reshards parameters in the post-forward."""
    if not handle:
        return
    # Do not free the root's parameters in the post-forward for `FULL_SHARD`
    # with the intention that they are immediately used for backward
    # computation (though this may not be true)
    free_unsharded_flat_param = (
        not state._is_root
        and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )
    _reshard(state, handle, free_unsharded_flat_param)


@no_type_check
def _root_pre_forward(
    state: _FSDPState,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    # [FUNCTION DOC] Root-only pre-forward logic. Runs lazy init. For a non-root state returns the
    # cast inputs (composable) or unchanged args/kwargs; for the root, restores buffers to
    # full precision when any handle forces it (or checks/restores low precision buffers
    # after leaving eval), waits for the default stream to finish pending work, resets
    # per-iteration state, and casts forward inputs to the param dtype if configured. Returns
    # (args, kwargs).
    # KNOWN ISSUE: annotated as returning None although it returns an (args, kwargs) tuple.
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._root_pre_forward"):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, "Expects a root FSDP to have been set")
        if not state._is_root:
            # Always cast forward inputs in the root of this local FSDP unit for mixed
            # precision, as this is where mixed precision could be configured.
            # This is more useful for auto wrapping that is recommended in composable path.
            # For manual wrapping, cast forward inputs on each local FSDP unit root will
            # increase some overhead, so not turned on for model wrapper path right now where
            # manual wrapping is more broadly used.
            if _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return args, kwargs

        # We cast buffers back to full precision if we're forcing full precision. Disjointly, we check if buffers
        # are in full precision and if we should cast them back to lower precision, which happens when
        # exiting eval() mode.
        handle = state._handle
        if handle:
            should_cast_buffers_to_full_prec = handle._force_full_precision
        else:
            # If the root has no handle (no managed parameters), then we fall
            # back to checking if any child wants to force full precision as a
            # workaround
            handles = traversal_utils._get_fsdp_handles(module)
            should_cast_buffers_to_full_prec = any(
                handle._force_full_precision for handle in handles
            )

        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(
                buffers=dict(module.named_buffers()).values(),
                buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()),
                device=state.compute_device,
            )
            # This flag is only set when we cast buffers to full precision, to avoid the
            # CPU overhead that can stem from retrieving all buffers and their types in the
            # following else branch.
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, "_needs_buffer_dtype_restore_check", False):
            # Check if buffers are in full precision and we need to cast them
            # back down.
            (
                buffers,
                buffer_dtypes_for_computation,
            ) = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any(
                    buffer.dtype != buffer_dtype_for_computation
                    for buffer, buffer_dtype_for_computation in zip(
                        buffers, buffer_dtypes_for_computation
                    )
                ):
                    # Assume we have to cast everything if there is one mismatch
                    _cast_buffers_to_dtype_and_device(
                        buffers, buffer_dtypes_for_computation, state.compute_device
                    )
            # We don't have to check this again until we cast buffers to full precision again.
            state._needs_buffer_dtype_restore_check = False

        if state.forward_prefetch:
            handles = [
                fsdp_state._handle
                for fsdp_state in state._all_fsdp_states
                if fsdp_state._handle
            ]
            for handle in handles:
                handle._needs_pre_forward_unshard = True
                handle._prefetched = False
        _wait_for_computation_stream(
            state._device_handle.current_stream(),
            state._unshard_stream,
            state._pre_unshard_stream,
        )
        _reset_flat_param_grad_info_if_needed(state._all_handles)

        # Prepares the forward inputs by moving them to ``compute_device``
        # TODO: Do not use the side stream for tensor copies for now; investigate
        # the perf with/without it.
        with torch.profiler.record_function("FullyShardedDataParallel._to_kwargs"):
            args_tuple, kwargs_tuple = _to_kwargs(
                args, kwargs, state.compute_device, False
            )
        args = args_tuple[0] if args_tuple else ()
        kwargs = kwargs_tuple[0] if kwargs_tuple else {}

        return _root_cast_forward_input(state, module, args, kwargs)


@no_type_check
def _root_cast_forward_input(
    state: _FSDPState, module: torch.nn.Module, args, kwargs
) -> tuple[Any, Any]:
    # [FUNCTION DOC] HELPER: Cast args and kwargs to the mixed-precision param dtype when the config
    # enables casting root forward inputs, the module is training (or full precision in eval
    # is off), and the handle is not forcing full precision. Returns (args, kwargs).
    # KNOWN ISSUE: the local flag named force_full_precision actually holds the inverse of the
    # handle's force-full-precision flag.
    if state._handle:
        force_full_precision = not state._handle._force_full_precision
    else:
        force_full_precision = True

    should_cast_forward_inputs = (
        (module.training or not state._use_full_prec_in_eval) and force_full_precision
    ) and state.mixed_precision.cast_root_forward_inputs

    if should_cast_forward_inputs:
        input_dtype: torch.dtype | None = state.mixed_precision.param_dtype
        args, kwargs = _cast_forward_inputs(input_dtype, *args, **kwargs)

    return args, kwargs


@no_type_check
def _pre_backward_hook(
    state: _FSDPState,
    module: nn.Module,
    handle: FlatParamHandle,
    grad,
    *unused: Any,
) -> Any:
    # [FUNCTION DOC] Pre-backward hook on forward outputs; returns grad unchanged. Skips if the
    # handle already ran it this backward. On the root it queues the final post-backward
    # callback once and resets grad info; otherwise asserts a valid training state. Sets the
    # training state; with a handle, marks BACKWARD_PRE, unshards if needed (unless
    # prefetched) and waits on the unshard stream, prefetches the next handle, prepares the
    # gradient for backward, and flags that the hook ran.
    """
    Prepares ``_handle`` 's ``FlatParameter`` s for gradient computation.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).
    """
    # Only run the pre-backward hook once per group of handles involved in the
    # same module forward computation
    if (
        handle
        and hasattr(handle, "_ran_pre_backward_hook")
        and handle._ran_pre_backward_hook
    ):
        return grad

    with torch.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
        # Queue the post-backward callback once for the root FSDP instance to
        # attach it to the outermost backward graph task so that it is called
        # after all backward calls complete
        if state._is_root and not state._post_backward_callback_queued:
            _register_post_backward_final_callback(state, module)
            _reset_flat_param_grad_info_if_needed(state._all_handles)
        elif handle:
            allowed_states = [TrainingState.IDLE]
            if _is_composable(state):
                allowed_states.append(TrainingState.FORWARD_BACKWARD)
            _assert_in_training_states(state, allowed_states)
        state.training_state = TrainingState.FORWARD_BACKWARD
        # Queueing the post-backward callback is the only logic that is not
        # per-handle in the pre-backward hook, so we can return early here if
        # there are no handles.
        if not handle:
            return grad
        handle._training_state = HandleTrainingState.BACKWARD_PRE

        if handle._needs_pre_backward_unshard:
            # If the handles have been prefetched, then there is no need to
            # call `_unshard()` again
            if not handle._prefetched:
                _unshard(
                    state,
                    handle,
                    state._unshard_stream,
                    state._pre_unshard_stream,
                )
            # Don't wait during trace
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                state._device_handle.current_stream().wait_stream(state._unshard_stream)

        # Set this to `False` to ensure that a mistargeted prefetch does not
        # actually unshard these handles
        handle._needs_pre_backward_unshard = False
        with torch.profiler.record_function(
            "FullyShardedDataParallel._pre_backward_prefetch"
        ):
            _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)
        handle.prepare_gradient_for_backward()
        handle._ran_pre_backward_hook = True
        return grad


@no_type_check
@torch.no_grad()
def _post_backward_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    flat_param,
    *unused: Any,
):
    # [FUNCTION DOC] Post-backward hook (no grad) for a handle's flat parameter. Marks that the hook
    # ran and the handle BACKWARD_POST; returns if there is no grad; raises RuntimeError on
    # double-backward grads. Reshards, then if gradients are not being synced only switches
    # to unsharded grad views and returns. Otherwise, in the post-backward stream, casts the
    # grad to the reduce dtype if needed and reduces it (reduce-scatter for sharded
    # strategies, all-reduce for no-shard) and records stream usage of the original grad.
    # Asserts valid training states.
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    _log_post_backward_hook(state, handle, logger)
    flat_param = handle.flat_param
    flat_param._post_backward_called = True
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # For multiple applications of reentrant AC across submodules sharing
        # the same `FlatParameter`, the post-backward hook may run multiple
        # times in one backward, in which case we permit the state to already
        # be in `BACKWARD_POST`.
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST

        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")

        _post_backward_reshard(state, handle)
        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return

        # Wait for all ops in the current stream (e.g. gradient computation) to
        # finish before reduce-scattering the gradient
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            state._post_backward_stream.wait_stream(
                state._device_handle.current_stream()
            )

        with state._device_handle.stream(state._post_backward_stream):
            autograd_computed_grad = flat_param.grad.data
            if (
                not _low_precision_hook_enabled(state)
                and flat_param.grad.dtype != handle._reduce_dtype
                # If we are forcing full precision but communicating grads
                # (i.e. model.eval() + full precision in eval was configured), don't downcast gradient.
                and not handle._force_full_precision
            ):
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
            if handle.uses_sharded_strategy:
                _reduce_grad(state, handle)
            else:
                _reduce_grad_no_shard(state, handle)
            # Since the unsharded gradient is produced in the computation
            # stream and consumed in the post-backward stream, inform the
            # caching allocator (before it goes out of scope)
            _no_dispatch_record_stream(
                autograd_computed_grad, state._post_backward_stream
            )


def _post_backward_reshard_only_hook(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
) -> None:
    # [FUNCTION DOC] Post-backward hook that only reshards, used when the pre-backward hook may not
    # have run (outputs not requiring grad): forces FORWARD_BACKWARD state, BACKWARD_POST
    # handle state, then reshards and prefetches.
    with torch.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook_reshard_only"
    ):
        # `_pre_backward_hook` may not get executed
        # if forward output does not require grad
        # overwrite IDLE state for post-backward prefetching
        state.training_state = TrainingState.FORWARD_BACKWARD
        handle._training_state = HandleTrainingState.BACKWARD_POST
        _post_backward_reshard(state, handle)


def _post_backward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
) -> None:
    # [FUNCTION DOC] HELPER: Reshard the handle (freeing the unsharded param per the free-in-backward
    # decision), then prefetch the next handle in backward mode.
    free_unsharded_flat_param = _should_free_in_backward(state, handle)
    _reshard(state, handle, free_unsharded_flat_param)

    # TODO: Post-backward prefetching does not support the multiple handles
    # per module case since the post-backward hook runs per handle, not per
    # group of handles.
    with torch.profiler.record_function(
        "FullyShardedDataParallel._post_backward_prefetch"
    ):
        _prefetch_handle(state, handle, _PrefetchMode.BACKWARD)


@no_type_check
def _should_free_in_backward(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> bool:
    # [FUNCTION DOC] HELPER: False for non-sharded strategies. Otherwise True if gradients are being
    # synced or the strategy reshards after forward (heuristic trading memory for speed
    # when accumulating without sync).
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
    if not handle.uses_sharded_strategy:
        return False
    # If not syncing gradients, then we do not free for strategies that do not
    # reshard after forward as a *heuristic* to tradeoff higher memory for
    # higher throughput.
    return (
        state._sync_gradients
        or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
    )


@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    # [FUNCTION DOC] HELPER: Reduce the unsharded grad for sharded strategies. Detaches the grad from
    # the flat param, builds padded input/output tensors, then either runs the default path
    # (pre-divide, reduce-scatter on the real or fake process group, and for hybrid sharding an
    # inter-node all-reduce on a separate stream plus post-divide) or calls the user comm
    # hook. Accumulates into the saved grad shard and runs the post-reduction callback.
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(
        state, unsharded_grad
    )
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = (
            handle._fake_process_group
            if handle._use_fake_reduce
            else state.process_group
        )
        dist.reduce_scatter_single(
            new_sharded_grad,
            padded_unsharded_grad,
            group=pg,
        )
        if uses_hybrid_sharded_strategy:
            # Don't wait during trace
            if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
                state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                # Since the new sharded gradient is produced in the post-
                # backward stream and consumed in the all-reduce stream,
                # inform the caching allocator
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(
                    state, handle, new_sharded_grad
                )
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(
            state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
        )
        # NOTE: HSDP variants do not support communication hook.
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _get_reduce_scatter_tensors(
    state: _FSDPState, unsharded_grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Chunk the grad by world size; return the grad padded to a multiple of
    # the chunk size (as the reduce-scatter input) and an empty tensor shaped like one chunk
    # (as the output).
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
    chunks = list(unsharded_grad.chunk(state.world_size))
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    padded_unsharded_grad = (
        F.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    )
    new_sharded_grad = torch.empty_like(chunks[0])  # padded
    return padded_unsharded_grad, new_sharded_grad


@no_type_check
def _accumulate_sharded_grad(
    state: _FSDPState,
    handle: FlatParamHandle,
    sharded_grad: torch.Tensor,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Cast the sharded grad to the param dtype, then add it into the flat
    # param's saved grad shard if present (after a compatibility check) or store it as the
    # saved shard. Returns the saved shard, which is the grad to offload.
    """
    Accumulates the reduce-scattered sharded gradient with any existing sharded
    gradient if needed, returning the gradient to offload (if CPU offloading is
    enabled).
    """
    flat_param = handle.flat_param
    _cast_grad_to_param_dtype(state, sharded_grad, flat_param)
    # Save the sharded gradient in `_saved_grad_shard` to support gradient
    # accumulation -- for multiple backwards, the gradient reductions may
    # happen in arbitrary order
    accumulate_grad = hasattr(flat_param, "_saved_grad_shard")
    if accumulate_grad:
        _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad_shard)
        flat_param._saved_grad_shard += sharded_grad
    else:
        flat_param._saved_grad_shard = sharded_grad
    grad_to_offload = flat_param._saved_grad_shard
    return grad_to_offload


@no_type_check
def _reduce_grad_no_shard(state: _FSDPState, handle: FlatParamHandle) -> None:
    # [FUNCTION DOC] HELPER: For NO_SHARD, all-reduce the flat param grad in place (with pre/post
    # division) or call the comm hook, cast to the param dtype unless low precision grads are
    # kept, then run the post-reduction callback with the grad data.
    """
    For no-shard, this runs gradient reduction (which directly covers any
    gradient accumulation implicitly) and the post-reduction callback.
    """
    flat_param = handle.flat_param
    if state._comm_hook is None:  # default path
        _div_if_needed(flat_param.grad, state._gradient_predivide_factor)
        dist.all_reduce(flat_param.grad, group=state.process_group)
        _div_if_needed(flat_param.grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, flat_param.grad)
    # For `NO_SHARD`, we can keep the low precision gradients by simply
    # omitting the cast altogether
    if not handle._keep_low_precision_grads:
        _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
    grad_to_offload = flat_param.grad.data
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _post_reduce_grad_callback(
    state: _FSDPState,
    handle: FlatParamHandle,
    # Additional arguments needed for the callback logic
    grad_to_offload: torch.Tensor,
):
    # [FUNCTION DOC] HELPER: After reduction, offload the grad to CPU if parameters are offloaded and
    # switch to sharded grad views if using original params.
    """
    This callback captures any logic to run after the gradient reduction
    finishes. Currently, this offloads the gradient to CPU if CPU offloading is
    enabled and uses sharded gradient views if ``use_orig_params=True``.
    """
    _offload_grad(state, handle, grad_to_offload)
    _post_backward_use_sharded_grad_views(handle)


@no_type_check
def _offload_grad(
    state: _FSDPState,
    handle: FlatParamHandle,
    grad_to_offload: torch.Tensor,
):
    # [FUNCTION DOC] HELPER: If params are offloaded, copy the grad into the flat param's CPU grad
    # buffer (non-blocking only for sharded strategies without in-backward optimizers) and
    # record stream usage for the allocator. No-op otherwise.
    if not handle._offload_params:
        return
    # Offload the gradient to CPU to ensure parameters and gradients are on the
    # same device as required by the optimizer
    # TODO: Investigate why `NO_SHARD` breaks correctness when using
    # `non_blocking=True` here.
    # TODO (rohan-varma): When CPU offload and optimizer overlap,
    # non_blocking=True won't work since the copy may have not finished before
    # the optimizer step executes on CPU. If we want to use non-blocking=True
    # here, we'll have to synchronize before using result on CPU.
    non_blocking = handle.uses_sharded_strategy and not handle._has_optim_in_backward
    handle.flat_param._cpu_grad.copy_(
        grad_to_offload.detach(), non_blocking=non_blocking
    )  # synchronized in the post-backward callback
    # Since the gradient being offloaded may have been produced in the
    # computation stream and is being consumed here in the post-backward
    # stream, inform the caching allocator
    _no_dispatch_record_stream(grad_to_offload.data, state._post_backward_stream)


@no_type_check
def _post_backward_use_sharded_grad_views(handle: FlatParamHandle):
    # [FUNCTION DOC] HELPER: No-op unless using original params. Resets the grad-is-none mask and
    # switches to sharded grad views. With optimizer-in-backward, prepares grads for the
    # optimizer, steps each in-backward optimizer of params that have a grad, zeroes grads,
    # resets flat param grad info, and drops the CPU grad buffer when offloading.
    # KNOWN ISSUE: the zero_grad call sits outside the inner optimizer loop, so it only applies
    # to the last optimizer of each parameter and relies on the loop variable leaking.
    if not handle._use_orig_params:
        return
    # Since the handle's `FlatParameter` completed its gradient computation, we
    # should reset the gradient noneness mask
    handle._reset_is_grad_none()
    # Delay using sharded gradient views until after the reduce-scatter instead
    # of immediately after resharding
    handle._use_sharded_grad_views()
    if handle._has_optim_in_backward:
        handle.prepare_gradient_for_optim()
        for orig_param in handle.flat_param._params:
            # Check for `None` gradient to filter parameters not in the rank
            if orig_param.grad is not None and hasattr(
                orig_param, "_in_backward_optimizers"
            ):
                # TODO (rohan-varma): For CPU offload, this unfortunately
                # operates on CPU because the parameters and gradients have
                # already been offloaded. We should run this on GPU after
                # refactoring.
                for optim in orig_param._in_backward_optimizers:
                    optim.step()

                optim.zero_grad(set_to_none=True)
        handle._reset_flat_param_grad_info_if_needed()
        if handle._offload_params:
            handle.flat_param._cpu_grad = None


def _div_if_needed(tensor: torch.Tensor, div_factor: float) -> None:
    # [FUNCTION DOC] HELPER: Divide the tensor in place by div_factor only when it is greater than 1.
    if div_factor > 1:
        tensor.div_(div_factor)


@no_type_check
def _cast_grad_to_param_dtype(
    state: _FSDPState,
    sharded_grad: torch.Tensor,
    param: FlatParameter,
):
    # [FUNCTION DOC] HELPER: Asserts FORWARD_BACKWARD state; unless a low precision comm hook is
    # enabled, cast the sharded grad data to the param dtype in place when dtypes differ,
    # and record stream usage for the allocator.
    """
    Casts ``sharded_grad`` back to the full parameter dtype so that the
    optimizer step runs with that dtype. This performs an actual cast if
    1. parameters were in reduced precision during the forward since then
    gradients would be in that reduced precision, or
    2. parameters were not in reduced precision but gradients were in
    reduced precision for communication.
    However, if a low precision communication hook is registered, then this
    dtype cast happens in the hook instead.
    """
    _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
    if not _low_precision_hook_enabled(state) and sharded_grad.dtype != param.dtype:
        low_prec_grad_data = sharded_grad.data
        sharded_grad.data = sharded_grad.data.to(dtype=param.dtype)
        # Since for `NO_SHARD`, the gradient is produced in the computation
        # stream and consumed here in the post-backward stream, inform the
        # caching allocator; for the sharded strategies, the gradient is
        # produced in the post-backward stream, so this `record_stream()`
        # should be a no-op
        _no_dispatch_record_stream(
            low_prec_grad_data, state._device_handle.current_stream()
        )


def _check_grad_to_accumulate(
    new_sharded_grad: torch.Tensor,
    accumulated_grad: torch.Tensor,
) -> None:
    # [FUNCTION DOC] HELPER: Assert (via _p_assert) that the new and accumulated sharded grads have
    # the same shape and device before accumulating.
    _p_assert(
        accumulated_grad.shape == new_sharded_grad.shape,
        "Shape mismatch when accumulating gradients: "
        f"existing gradient shape={accumulated_grad.shape} "
        f"new gradient shape={new_sharded_grad.shape}",
    )
    _p_assert(
        accumulated_grad.device == new_sharded_grad.device,
        "Device mismatch when accumulating gradients: "
        f"existing gradient device={accumulated_grad.device} "
        f"new gradient device={new_sharded_grad.device}",
    )


@no_type_check
def _low_precision_hook_enabled(state: _FSDPState) -> bool:
    # [FUNCTION DOC] HELPER: True if the state's comm hook is one of the known low precision hooks.
    return state._comm_hook in LOW_PRECISION_HOOKS


@no_type_check
@torch.no_grad()
def _post_backward_final_callback(
    state: _FSDPState,
    module: nn.Module,
):
    # [FUNCTION DOC] Final callback queued on the autograd engine, run on the root at the end of the
    # whole backward (no grad). If syncing grads, makes the current stream wait on the
    # post-backward and all-reduce streams and, with param offload, synchronizes. Advances
    # exec order data, then for every FSDP state reshards leftovers, finalizes params, resets
    # states and per-iteration handle flags. Finally clears the callback-queued flag.
    # Asserts the state is the root.
    """
    This waits for the post-backward to finish and performs some final cleanup.
    This runs at the end of the entire backward pass and should only be called
    on the root FSDP instance.
    """
    _p_assert(
        state._is_root,
        "The post-backward callback should only be called on the root FSDP instance",
    )
    root_state = state

    if root_state._sync_gradients:
        current_stream = state._device_handle.current_stream()
        # TODO (rohan-varma): this also waits for the overlapped optimizer step to finish
        # since it currently runs in the post-backward stream. That can be
        # pushed to the next forward if run in a different stream
        current_stream.wait_stream(root_state._post_backward_stream)
        if root_state._all_reduce_stream is not current_stream:  # uses HSDP
            current_stream.wait_stream(root_state._all_reduce_stream)
        if root_state.cpu_offload.offload_params:
            # Wait for non-blocking GPU -> CPU sharded gradient copies from the
            # post-backward hooks to finish explicitly since CPU gradients do
            # not automatically synchronize with the GPU
            state._device_handle.current_stream().synchronize()
    root_state._exec_order_data.next_iter()

    for fsdp_state in state._all_fsdp_states:
        _catch_all_reshard(fsdp_state)
        _finalize_params(fsdp_state)
        fsdp_state.training_state = TrainingState.IDLE
        handle = fsdp_state._handle
        if handle:
            handle._ran_pre_backward_hook = False
            handle._needs_pre_backward_unshard = False
            handle._post_forward_index = None
            handle._training_state = HandleTrainingState.IDLE
            handle._prefetched = False
    # Reset for cases like one forward and multiple backwards
    root_state._post_backward_callback_queued = False


@no_type_check
def _catch_all_reshard(
    state: _FSDPState,
) -> None:
    # [FUNCTION DOC] HELPER: Reshard the state's handle if its flat param still points to unsharded
    # data (or sharded views were skipped), using the free-in-backward decision. Errors are
    # reported through _p_assert (non-raising) and then re-raised.
    # KNOWN ISSUE: the already-resharded check compares data pointers and is noted as brittle.
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not used in the loss computation corresponding to this backward pass.
    """
    # Wrap with a try-except to provide a more informative traceback if an
    # error is raised
    try:
        if state._handle:
            # TODO: This already-resharded check is brittle:
            # https://github.com/pytorch/pytorch/issues/83956
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._local_shard.data_ptr()
                # If FSDP skipped using sharded views, then the flat parameter
                # still points to the sharded data, so we need to reshard to
                # use sharded views
                and not state._handle._skipped_use_sharded_views
            )
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        raise e


@no_type_check
def _finalize_params(
    state: _FSDPState,
) -> None:
    # [FUNCTION DOC] HELPER: Per-iteration cleanup for a state's flat param (no-op without handle).
    # Removes the registered post-backward hook (handle under compile, hook state otherwise,
    # validating its length). If the flat param requires grad: when not syncing gradients keeps
    # accumulation state and returns; else prepares the gradient for the optimizer (unless the
    # optimizer ran in backward), asserts the post-backward-called flag exists, and clears it.
    """Finalizes the parameters before the next iteration."""
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        if hasattr(flat_param, "_post_backward_hook_handle"):
            pbhs_handle = flat_param._post_backward_hook_handle
            pbhs_handle.remove()
            del flat_param._post_backward_hook_handle
    else:
        if hasattr(flat_param, "_post_backward_hook_state"):
            post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
            expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
            _p_assert(
                post_backward_hook_state_len == expected_post_backward_hook_state_len,
                f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
            )
            flat_param._post_backward_hook_state[-1].remove()
            delattr(flat_param, "_post_backward_hook_state")
    if flat_param.requires_grad:
        if not state._sync_gradients:
            # Preserve the gradient accumulation state if not synchronizing
            # gradients: `.grad` remains the unsharded gradient  from prior
            # `no_sync()` iterations, and `_saved_grad_shard` remains the
            # sharded gradient from the last synchronized iteration
            return
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        flat_param._post_backward_called = False


@no_type_check
def _prefetch_handle(
    state: _FSDPState,
    current_handle: FlatParamHandle | None,
    prefetch_mode: _PrefetchMode,
) -> None:
    # [FUNCTION DOC] Prefetch (unshard early without synchronization) the next handle chosen from
    # the execution order. No-op if there is no current handle or nothing to prefetch. Temporarily
    # sets the target's training state to match the mode (raises ValueError for an unknown
    # mode), unshards it, restores the state, and marks it prefetched.
    """
    Prefetches the next handles if needed (without synchronization). An empty
    handles key cannot prefetch.
    """
    if not current_handle:
        return
    handle = _get_handle_to_prefetch(state, current_handle)
    if not handle:
        return
    # Temporarily emulate the training state while calling `_unshard` to
    # ensure the correct `as_params` for `_use_unsharded_views()`
    prev_training_state = handle._training_state
    if prefetch_mode == _PrefetchMode.BACKWARD:
        handle._training_state = HandleTrainingState.BACKWARD_PRE
    elif prefetch_mode == _PrefetchMode.FORWARD:
        handle._training_state = HandleTrainingState.FORWARD
    else:
        raise ValueError(f"Invalid prefetch mode on rank {state.rank}: {prefetch_mode}")
    # Prefetch the next set of handles without synchronizing to allow
    # the sync to happen as late as possible to maximize overlap
    _unshard(state, handle, state._unshard_stream, state._pre_unshard_stream)
    handle._training_state = prev_training_state
    handle._prefetched = True


@no_type_check
def _get_handle_to_prefetch(
    state: _FSDPState,
    current_handle: FlatParamHandle,
) -> FlatParamHandle:
    # [FUNCTION DOC] HELPER: Choose the handle to prefetch next, or None. Asserts the current handle is
    # in a backward or forward state. In BACKWARD_PRE or BACKWARD_POST matching the configured
    # backward prefetch policy, or in FORWARD with forward prefetch on, asks the execution
    # order data for the candidate and accepts it only if it still needs unsharding and has
    # not been prefetched.
    # KNOWN ISSUE: the docstring says a list is returned, but a single handle or None is.
    """
    Returns a :class:`list` of the handles keys to prefetch for the next
    module(s), where ``current_handle`` represents the current module.

    "Prefetching" refers to running the unshard logic early (without
    synchronization), and the "next" modules depend on the recorded execution
    order and the current training state.
    """
    training_state = _get_training_state(current_handle)
    valid_training_states = (
        HandleTrainingState.BACKWARD_PRE,
        HandleTrainingState.BACKWARD_POST,
        HandleTrainingState.FORWARD,
    )
    _p_assert(
        training_state in valid_training_states,
        f"Prefetching is only supported in {valid_training_states} but "
        f"currently in {training_state}",
    )
    eod = state._exec_order_data
    target_handle: FlatParamHandle | None = None
    if (
        training_state == HandleTrainingState.BACKWARD_PRE
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
    ) or (
        training_state == HandleTrainingState.BACKWARD_POST
        and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
    ):
        target_handle_candidate = eod.get_handle_to_backward_prefetch(current_handle)
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_backward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None
    elif training_state == HandleTrainingState.FORWARD and state.forward_prefetch:
        target_handle_candidate = eod.get_handle_to_forward_prefetch(current_handle)
        if (
            target_handle_candidate
            and target_handle_candidate._needs_pre_forward_unshard
            and not target_handle_candidate._prefetched
        ):
            target_handle = target_handle_candidate
        else:
            target_handle = None

    return target_handle


def _get_training_state(
    handle: FlatParamHandle,
) -> HandleTrainingState:
    # [FUNCTION DOC] HELPER: Return the handle's training state; asserts the handle is non-empty.
    """Returns the training state of the handles in ``handle``."""
    _p_assert(handle, "Expects a non-empty handle")
    return handle._training_state


@no_type_check
def _register_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    # [FUNCTION DOC] Remove any previously registered pre-forward hooks of the state, then register a
    # prepended forward pre-hook (with kwargs) on module that runs the pre-forward logic with
    # the module's handle and the pre-forward unshard function. Mutates state's handle list.
    """
    Registers a pre-forward hook on ``module``.
    """
    for forward_handle in state._pre_forward_handles:
        forward_handle.remove()
    state._pre_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(
        _pre_forward, state, module_param_handle, _pre_forward_unshard
    )
    state._pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


@no_type_check
def _register_post_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    # [FUNCTION DOC] Remove any previously registered post-forward hooks of the state, then register
    # a forward hook on module running the post-forward logic with its handle and the
    # post-forward reshard function (registered even without a handle).
    """
    Registers a post-forward hook on ``module``. Even if the module has no
    handles, we should register the hook since it will register the module's
    pre-backward hook.
    """
    for forward_handle in state._post_forward_handles:
        forward_handle.remove()
    state._post_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(
        _post_forward,
        state,
        module_param_handle,
        _post_forward_reshard,
    )
    state._post_forward_handles.append(module.register_forward_hook(hook))


@no_type_check
def _register_root_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
):
    # [FUNCTION DOC] Remove previously registered root pre-forward hooks of the state, then register
    # a prepended forward pre-hook (with kwargs) on module that runs the root pre-forward logic.
    """
    Registers root pre-forward hook on ``module``, which should be the local
    FSDP root.

    NOTE: For the current composable FSDP design, we have each application of
    ``fully_shard()`` to a module to indicate that that module is the local
    FSDP root. We may remove this assumption in the future, in which case we
    will need to register this root pre-forward hook on any candidate module
    that may be the local FSDP root.
    """
    for forward_handle in state._root_pre_forward_handles:
        forward_handle.remove()
    state._root_pre_forward_handles.clear()
    hook = functools.partial(_root_pre_forward, state)
    state._root_pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


@no_type_check
def _register_pre_backward_hooks(
    state: _FSDPState,
    module: nn.Module,
    outputs: Any,
    handle: FlatParamHandle,
) -> None:
    # [FUNCTION DOC] Register pre-backward hooks on every grad-requiring tensor in outputs and return
    # outputs (unchanged when grad is disabled). On the root it clears the callback-queued flag;
    # resets the handle's pre-backward flags and sets the needs-unshard flag when a hook is
    # registered.
    # KNOWN ISSUE: annotated as returning None although it returns the outputs.
    """
    Registers pre-backward hooks on the tensors that require gradients in the
    forward pass outputs ``outputs``, which were computed using the
    ``FlatParameter`` s of ``handles``.

    Args:
        module (nn.Module): Fully sharded module (see [Note: Fully Sharded
            Module]).

    Returns:
        Forward pass outputs with pre-backward hooks registered to tensors that
        require gradients.
    """
    # If there is no gradient computation, then there is no need for
    # pre-backward logic
    if not torch.is_grad_enabled():
        return outputs
    if state._is_root:
        state._post_backward_callback_queued = False  # only defined on the root

    if handle:
        handle._needs_pre_backward_unshard = False
        # Since these handles' `FlatParameter`s participated in a forward, we
        # conservatively assume that they will be used in the backward
        handle._ran_pre_backward_hook = False

    def _register_hook(t: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: If the tensor requires grad, register an unserializable pre-backward
        # hook on it and flag the handle as needing a pre-backward unshard. Returns the tensor.
        if t.requires_grad:
            t.register_hook(
                torch.utils.hooks.unserializable_hook(
                    functools.partial(_pre_backward_hook, state, module, handle)
                )
            )
            if handle:
                handle._needs_pre_backward_unshard = True
        return t

    return _apply_to_tensors(_register_hook, outputs)


def _register_post_backward_hook(
    state: _FSDPState,
    handle: FlatParamHandle | None,
) -> None:
    # [FUNCTION DOC] Register the post-backward hook for the handle's flat param once (first forward).
    # No-op without grad mode, handle, or if already registered or not requiring grad.
    # Under torch.compile uses a post-accumulate-grad hook handle; otherwise finds the
    # AccumulateGrad node via an expand_as view and hooks it, storing the node and hook
    # on the flat param. Raises AssertionError if the grad function or node is missing.
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    if not handle:
        return
    flat_param = handle.flat_param

    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        already_registered = hasattr(flat_param, "_post_backward_hook_handle")
        if already_registered or not flat_param.requires_grad:
            return
        hook = functools.partial(_post_backward_hook, state, handle)
        hook_handle = flat_param.register_post_accumulate_grad_hook(hook)
        flat_param._post_backward_hook_handle = hook_handle  # type: ignore[attr-defined]
    else:
        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            return
        # Get the `AccumulateGrad` object
        temp_flat_param = flat_param.expand_as(flat_param)
        _p_assert(
            temp_flat_param.grad_fn is not None,
            "The `grad_fn` is needed to access the `AccumulateGrad` and "
            "register the post-backward hook",
        )
        acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
        if acc_grad is None:
            raise AssertionError("Expected acc_grad to be set")
        hook_handle = acc_grad.register_hook(
            functools.partial(_post_backward_hook, state, handle)
        )
        flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]


def _register_post_backward_reshard_only_hook(
    state: _FSDPState,
    handle: FlatParamHandle | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    # [FUNCTION DOC] For a flat param that does not require grad, register a multi-grad hook over the
    # grad-requiring tensor inputs (from args and kwargs) that runs the reshard-only hook.
    # No-op without grad mode, handle, or if a hook is already registered or the flat param
    # requires grad. Stores the hook handle on the flat param (as hook handle under compile,
    # else as a one-element hook state).
    # KNOWN ISSUE: the later check that the input tensor list is set can never fail.
    """
    Registers post-backward hooks to reshard flat parameters that do not
    require gradient. We register these using multi-post-grad hooks on the
    input activations to ensure that all gradients that may depend on the
    parameters have been computed before resharding.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    # Construct `inp_tensors` lazily to avoid CPU overhead in typical case
    # where each flat parameter requires gradient
    inp_tensors: list[torch.Tensor] | None = None
    if not handle:
        return
    flat_param = handle.flat_param

    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        already_registered = hasattr(flat_param, "_post_backward_hook_handle")
    else:
        already_registered = hasattr(flat_param, "_post_backward_hook_state")

    if already_registered or flat_param.requires_grad:
        return
    if inp_tensors is None:
        args_flat = pytree.arg_tree_leaves(*args, **kwargs)
        inp_tensors = [
            obj for obj in args_flat if torch.is_tensor(obj) and obj.requires_grad
        ]
    if inp_tensors is None:
        raise AssertionError("Expected inp_tensors to be set")
    hook_handle = register_multi_grad_hook(
        inp_tensors, functools.partial(_post_backward_reshard_only_hook, state, handle)
    )
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        flat_param._post_backward_hook_handle = hook_handle  # type: ignore[attr-defined, assignment]
    else:
        flat_param._post_backward_hook_state = (hook_handle,)  # type: ignore[attr-defined, assignment]


@no_type_check
def _register_post_backward_final_callback(
    state: _FSDPState, module: nn.Module
) -> None:
    # [FUNCTION DOC] Queue the final post-backward callback on the autograd engine once (root only,
    # skipped under torch.compile) and set the queued flag. No-op if already queued. Asserts
    # root and IDLE training state.
    """
    Registers the post-backward final callback that runs at the end of the
    backward pass. This should be called from the root FSDP instance at the
    beginning of the pre-backward.
    """
    _p_assert(
        state._is_root,
        "Only the root FSDP instance should register the post-backward callback",
    )
    if state._post_backward_callback_queued:
        return
    _assert_in_training_states(state, [TrainingState.IDLE])
    # Trace does not need this callback
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        state._post_backward_callback_queued = True
        Variable._execution_engine.queue_callback(
            functools.partial(_post_backward_final_callback, state, module)
        )


def _wait_for_computation_stream(
    computation_stream: torch.Stream,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
):
    # [FUNCTION DOC] Make the unshard and pre-unshard streams wait for the computation stream
    # (skipped under torch.compile tracing).
    """
    Has the unshard and pre-unshard streams wait for the computation stream.
    For example, this should be called in the FSDP root's pre-forward to
    respect optimizer step computation.
    """
    # Tracing does not need to wait
    if torch.distributed._functional_collectives.is_torchdynamo_compiling():
        return
    unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]
    # Having the pre-all-gather stream wait for the current stream even if we
    # do not leverage the pre-all-gather stream is tolerable since this only
    # runs once per iteration
    pre_unshard_stream.wait_stream(computation_stream)  # type: ignore[attr-defined]


def _reset_flat_param_grad_info_if_needed(
    handles: list[FlatParamHandle],
):
    # [FUNCTION DOC] For each handle (a single handle is wrapped into a list) that uses original
    # params, reset its flat param grad info to free gradient memory early.
    """
    Clears the original parameters' gradients if needed. This method's CPU
    overhead is minimal, so we may call it throughout FSDP methods, which serve
    as callsites to free the gradient memory earlier.
    """
    if not isinstance(handles, list):
        handles = [handles]
    for handle in handles:
        if handle._use_orig_params:
            handle._reset_flat_param_grad_info_if_needed()


@no_type_check
def _get_buffers_and_dtypes_for_computation(
    state: _FSDPState,
    root_module: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.dtype | None]]:
    # [FUNCTION DOC] HELPER: Collect unique, non-ignored buffers in the tree with their compute
    # dtypes (the owning state's buffer dtype, or None). States are traversed bottom-up so the
    # innermost owning state's setting wins. Returns parallel lists; asserts root state.
    """
    Returns all buffers in the module tree rooted at ``root_module`` and a
    corresponding list of the buffer dtypes for computation. Each buffer dtype
    is either ``None`` if buffer mixed precision is not enabled or the buffer
    low precision dtype otherwise.
    """
    _p_assert(state._is_root, "Expects the root to cast buffers")
    buffers: list[torch.Tensor] = []
    buffer_dtypes: list[torch.dtype | None] = []
    visited_buffers: set[torch.Tensor] = set()
    # Traverse the FSDP states bottom-up so that we prefer the owning FSDP
    # instance's mixed precision setting for each buffer
    fsdp_states, fsdp_modules = traversal_utils._get_fsdp_states_with_modules(
        root_module
    )
    for fsdp_state, fsdp_module in zip(reversed(fsdp_states), reversed(fsdp_modules)):
        for buffer_name, buffer in fsdp_module.named_buffers():
            if buffer in visited_buffers:
                continue
            visited_buffers.add(buffer)
            if clean_tensor_name(buffer_name) in fsdp_state._ignored_buffer_names:
                continue
            buffers.append(buffer)
            buffer_dtypes.append(fsdp_state.mixed_precision.buffer_dtype)
    if len(buffers) != len(buffer_dtypes):
        raise AssertionError(
            f"Expected buffers and buffer_dtypes to have the same length, got {len(buffers)} and {len(buffer_dtypes)}"
        )
    return buffers, buffer_dtypes


@no_type_check
def _get_orig_buffer_dtypes(
    state: _FSDPState,
    buffer_names: list[str],
) -> list[torch.dtype]:
    # [FUNCTION DOC] HELPER: Look up each buffer name's original dtype in the state's precomputed
    # mapping; asserts (via _p_assert) that every name is present. Returns the dtype list.
    """
    Returns the original buffer types of the given buffer names.
    """
    buffer_dtypes: list[torch.dtype] = []
    for buffer_name in buffer_names:
        _p_assert(
            buffer_name in state._buffer_name_to_orig_dtype,
            f"{buffer_name} is missing from pre-computed dict on rank "
            f"{state.rank}, which only has keys "
            f"{state._buffer_name_to_orig_dtype.keys()}",
        )
        buffer_dtypes.append(state._buffer_name_to_orig_dtype[buffer_name])
    return buffer_dtypes


def _cast_buffers_to_dtype_and_device(
    buffers: list[torch.Tensor],
    buffer_dtypes: list[torch.dtype | None],
    device: torch.device,
) -> None:
    # [FUNCTION DOC] Move each buffer to device in place (by replacing its data), also casting to its
    # paired dtype when the buffer is floating point and the dtype is not None. Asserts
    # the two lists have equal length.
    # KNOWN ISSUE: the length message formats the length of buffer_dtypes, which would fail if
    # buffer_dtypes were None, though the assert allows None (the zip would also then fail).
    """
    Casts ``buffers`` to the dtypes given by ``buffer_dtypes`` and moves them
    to ``device``. If an element in ``buffer_dtypes`` is ``None``, then the
    corresponding buffer is only moved to ``device``.
    """
    _p_assert(
        buffer_dtypes is None or len(buffers) == len(buffer_dtypes),
        f"Expects `buffers` and `buffer_dtypes` to have the same length if "
        f"`buffer_dtypes` is specified but got {len(buffers)} and "
        f"{len(buffer_dtypes)}",
    )
    for buffer, buffer_dtype in zip(buffers, buffer_dtypes):
        if not torch.is_floating_point(buffer) or buffer_dtype is None:
            buffer.data = buffer.to(device=device)
        else:
            buffer.data = buffer.to(device=device, dtype=buffer_dtype)
