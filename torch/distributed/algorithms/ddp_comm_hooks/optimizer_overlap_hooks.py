# mypy: allow-untyped-defs
# [MODULE DOC] Support for running a functional optimizer step right after DDP gradient communication
# (optimizer overlap): the hook-state holder, the optimizer-in-backward hook factory and a hook wrapper.
# Exports nothing publicly (empty __all__).
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, no_type_check

import torch
import torch.distributed as dist
from torch.autograd import Variable


__all__: list[str] = []

_FUNCTIONAL_OPTIM_STEP_METHOD_NAME = "step_param"


class _OptimizerHookState:
# [CLASS DOC] HELPER: Slots-based state with the functional optimizer (must have a step_param method) and an
# optional set of parameters to optimize. params_to_optimize is only assigned when params is not None.
    """
    Holds state for running optimizer in-line after DDP communication hook.

    Currently contains only optimizer class which must have a method `step_param`.
    """

    __slots__ = ["functional_optimizer", "params_to_optimize"]

    def __init__(self, functional_optim, params=None):
        # [FUNCTION DOC] HELPER: Store the functional optimizer, validate it has step_param, record params to optimize.
        self.functional_optimizer = functional_optim
        self._check_valid_functional_optim()
        self._set_params_to_optimize(params)

    def _set_params_to_optimize(self, params):
        # [FUNCTION DOC] HELPER: If params is not None, store it as a set; otherwise leave the attribute unset.
        if params is not None:
            self.params_to_optimize = set(params)

    def _check_valid_functional_optim(self):
        # [FUNCTION DOC] HELPER: Raise ValueError if the functional optimizer lacks a step_param method.
        if not hasattr(self.functional_optimizer, _FUNCTIONAL_OPTIM_STEP_METHOD_NAME):
            raise ValueError(
                f"Class {type(self.functional_optimizer)} must implement method "
                f"{_FUNCTIONAL_OPTIM_STEP_METHOD_NAME}."
            )


@dataclass
class _OptimInBackwardHookState:
# [CLASS DOC] HELPER: Dataclass holding the stream on which optimizer steps run in backward and a flag
# telling whether the end-of-backward wait callback was already enqueued.
    optim_stream: torch.Stream
    wait_for_optim_stream_enqueued: bool


@no_type_check
def _apply_optim_in_backward_hook(
    gradient_is_bucket_view: bool,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Factory that returns a DDP comm hook (a partial with its own stream state) which
    # allreduces a bucket, divides by group size, and runs in-backward optimizers attached to the bucket's
    # parameters on a side stream. If gradient_is_bucket_view is false, each parameter grad is first set to the
    # bucket gradient. Copies the hook name and qualname onto the partial for DDP logging.
    r"""
    Register hook to apply the optimizer in backward.

    If torch.distributed.optim._apply_optimizer_in_backward is used to overlap
    optimizer with backward pass, DDP will run the below hook to run optimizer
    step for parameters after gradient communication has taken place.
    """
    optim_in_bwd_state = _OptimInBackwardHookState(
        optim_stream=torch.Stream(),
        wait_for_optim_stream_enqueued=False,
    )

    def apply_optim_in_backward_hook(
        hook_state: Any,
        bucket: dist.GradBucket,
        optim_stream_state,
    ) -> torch.futures.Future[torch.Tensor]:
        # [FUNCTION DOC] HELPER: The comm hook body: runs the reducer allreduce on the optimizer stream, averages
        # the buffer, steps each parameter's in-backward optimizers, returns a completed future of the buffer, and
        # enqueues (once per backward) a callback that syncs the stream and clears grads of managed params.
        # KNOWN ISSUE: TODO in code notes the division factor may be wrong with the join hook.
        # Run original hook
        ddp_weakref = hook_state
        ddp_inst = ddp_weakref()
        reducer, process_group = ddp_inst.reducer, ddp_inst.process_group
        fut = reducer._run_allreduce_hook(bucket)
        optimizer_stream = optim_stream_state.optim_stream
        with optimizer_stream:
            fut.wait()
            # Apply gradient division since C++ side only allreduces and does
            # not average. TODO: (rohan-varma) the div factor may be different
            # when running with join hook
            bucket.buffer().div_(process_group.size())
            model_params = bucket.parameters()
            grads = bucket.gradients()
            # TODO (rohan-varma): upcast as needed for DDP mixed precision,
            # once optimizer in backward + DDP mixed precision is supported.
            for p, g in zip(model_params, grads):
                if hasattr(p, "_in_backward_optimizers"):
                    # Note: need to set grad to the bucket's grad, because
                    # running allreduce results in the bucket's grad being
                    # reduced, but not grad field.
                    if not gradient_is_bucket_view:
                        p.grad = g
                    for optim in p._in_backward_optimizers:
                        optim.step()

        # Need to return a Future[Tensor] to obey comm hook API contract.
        ret_fut = torch.futures.Future()
        ret_fut.set_result(bucket.buffer())

        # enqueue a callback to wait for this optimizer stream at the end of
        # backward and set all DDP managed grads to None.
        def wait_for_optim_stream_callback():
            # [FUNCTION DOC] HELPER: End-of-backward callback: current stream waits on the optimizer stream, grads of
            # DDP parameters with in-backward optimizers are set to None, and the enqueued flag is reset.
            torch.accelerator.current_stream().wait_stream(
                optim_stream_state.optim_stream
            )
            # Set DDP managed grads to None
            for param in ddp_inst._get_data_parallel_params(ddp_inst.module):
                if hasattr(param, "_in_backward_optimizers"):
                    param.grad = None

            # reset for the next backward pass
            optim_stream_state.wait_for_optim_stream_enqueued = False

        if not optim_stream_state.wait_for_optim_stream_enqueued:
            Variable._execution_engine.queue_callback(wait_for_optim_stream_callback)
            # mark that the callback is enqueued
            optim_stream_state.wait_for_optim_stream_enqueued = True

        return ret_fut

    comm_hook = partial(
        apply_optim_in_backward_hook, optim_stream_state=optim_in_bwd_state
    )
    # These are needed for DDP's logging of comm hooks
    comm_hook.__name__ = apply_optim_in_backward_hook.__name__
    comm_hook.__qualname__ = apply_optim_in_backward_hook.__qualname__

    return comm_hook


def _hook_then_optimizer(
    hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]],
    optimizer_state: _OptimizerHookState,
) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Return a wrapper hook that runs the given hook and, once its future completes,
    # applies the functional optimizer's step_param to each (param, grad) pair in the bucket (restricted to
    # params_to_optimize when set). Mutates parameters; the resulting future yields the bucket buffer.
    r"""Run optimizer in a functional fashion after DDP communication hook."""
    has_set_params = (
        hasattr(optimizer_state, "params_to_optimize")
        and optimizer_state.params_to_optimize is not None
    )

    def hook_then_optimizer_wrapper(
        hook_state, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Run the wrapped hook and chain the optimizer step onto its future; returns that future.
        # Run original hook
        fut = hook(hook_state, bucket)

        def optimizer_step(fut):
            # [FUNCTION DOC] HELPER: Apply the optimizer step per parameter after communication; returns the bucket buffer.
            gradient_tensors = bucket.gradients()
            model_params = bucket.parameters()
            for grad_tensor, model_param in zip(gradient_tensors, model_params):
                if (
                    not has_set_params
                    or model_param in optimizer_state.params_to_optimize
                ):
                    optimizer_state.functional_optimizer.step_param(
                        model_param,
                        grad_tensor,
                    )
            return bucket.buffer()

        return fut.then(optimizer_step)

    return hook_then_optimizer_wrapper
