# mypy: allow-untyped-defs
# [MODULE DOC] Composable (module-structure-preserving) activation checkpointing: defines
# the checkpoint API, built with the contract decorator, which installs forward pre/post
# hooks that run non-reentrant checkpointing from torch.utils.checkpoint.
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    _checkpoint_without_reentrant_generator_impl,
    _DEFAULT_DETERMINISM_MODE,
)

from .contract import _State, contract


@contextmanager
def _no_hook(module: nn.Module, user_ctx: AbstractContextManager | None = None):
    # [FUNCTION DOC] HELPER: Context manager used during recomputation. Enters the optional
    # user context, then temporarily sets the module's checkpoint enable_hook flag to False
    # (so hooks do not recurse) and restores the previous value on exit, even on error.
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """

    with user_ctx if user_ctx else nullcontext():
        orig_enable_hook = checkpoint.state(module).enable_hook
        checkpoint.state(module).enable_hook = False
        try:
            yield
        finally:
            checkpoint.state(module).enable_hook = orig_enable_hook


class _CheckpointState(_State):
    # [CLASS DOC] HELPER: Per-module state for composable checkpoint. enable_hook (default
    # False) gates the hooks; _ac_generator holds the live non-reentrant checkpoint
    # generator between the forward pre-hook and forward hook, or None.
    enable_hook: bool = False
    _ac_generator: Generator[None, None, None] | None


@contract(_CheckpointState)
def checkpoint(module: nn.Module, **kwargs) -> nn.Module:
    # [FUNCTION DOC] Applies activation checkpointing to module in place and returns it.
    # Keyword options popped from kwargs: use_reentrant (True raises NotImplementedError),
    # preserve_rng_state (default True), context_fn, determinism_check, debug, early_stop;
    # any leftover keyword raises ValueError. Logs API usage, sets the module's state
    # enable_hook to True, and registers a forward pre-hook (starts a checkpoint generator
    # when hooks are enabled) and an always-called, prepended forward hook (exhausts the
    # generator, raising RuntimeError if it is not exhausted, then clears it).
    r"""
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.

    Args:
        module (nn.Module): the target model or sub-module to apply activation
            checkpointing.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> model = MyModel()
        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1
        >>> model(torch.zeros(2, 10)).sum().backward()

    """
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    use_reentrant = kwargs.pop("use_reentrant", False)
    if use_reentrant:
        raise NotImplementedError(
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead."
        )
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    user_context_fns = kwargs.pop("context_fn", None)
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    debug = kwargs.pop("debug", False)
    early_stop = kwargs.pop("early_stop", True)

    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def forward_pre_hook(
        module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        # [FUNCTION DOC] HELPER: Forward pre-hook. When enabled for the module, creates the
        # non-reentrant checkpoint generator, stores it in the module state and advances it
        # once so the forward runs under checkpointing.
        if checkpoint.state(module).enable_hook:

            def context_fns():
                # [FUNCTION DOC] HELPER: Returns the pair of forward and recompute contexts;
                # uses the user's context_fn pair if given (wrapping the recompute one with
                # the hook-disabling context) else a null context and hook-disabling context.
                if user_context_fns is not None:
                    ctx1, ctx2 = user_context_fns()
                    return ctx1, _no_hook(module, ctx2)
                else:
                    return nullcontext(), _no_hook(module)

            gen = _checkpoint_without_reentrant_generator_impl(
                module,
                args,
                kwargs,
                preserve_rng_state=preserve_rng_state,
                context_fn=context_fns,
                determinism_check=determinism_check,
                debug=debug,
                early_stop=early_stop,
            )
            checkpoint.state(module)._ac_generator = gen
            next(gen)

    def forward_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        # [FUNCTION DOC] HELPER: Forward post-hook. If enabled, advances the stored generator
        # to completion (raises AssertionError if none stored, RuntimeError if it yields again).
        # Always clears the stored generator. Returns None (output is left unchanged).
        if checkpoint.state(module).enable_hook:
            try:
                gen = checkpoint.state(module)._ac_generator
                if gen is None:
                    raise AssertionError
                next(gen)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!"
                )

        #  Ensure that we no longer hold on to the generator. always_call=True helps ensure we
        # clear this even in the case of exception in fwd pass.
        checkpoint.state(module)._ac_generator = None

    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module
