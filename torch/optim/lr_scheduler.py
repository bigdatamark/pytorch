# mypy: allow-untyped-defs
# [MODULE DOC] Learning rate schedulers that adjust the lr entry of an optimizer's param
# groups over training.
# Main contents: the LRScheduler base class (chainable, recursive-form get_lr plus optional
# closed-form variant), many concrete schedulers (LambdaLR, MultiplicativeLR, StepLR,
# MultiStepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR,
# CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR), the composition schedulers
# SequentialLR and ChainedScheduler, and the metric-driven ReduceLROnPlateau.
# Also holds small private helpers for copying lr values without aliasing tensors and two
# context managers that flag state on a scheduler.
# Import-time side effects: none beyond defining names; the only public export list is
# __all__. EPOCH_DEPRECATION_WARNING is the message used when step is given an explicit
# epoch.
r"""Learning Rate Scheduler."""

from __future__ import annotations

import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from functools import partial, wraps
from typing import Any, cast, Literal, SupportsFloat, TYPE_CHECKING, TypedDict
from typing_extensions import override, Self
from weakref import ref

from torch import inf, Tensor

from .optimizer import _to_scalar, Optimizer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


__all__ = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
    "LRScheduler",
]

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


def _format_param(name: str, optimizer: Optimizer, param):
    # [FUNCTION DOC] HELPER: Return a list with one value per optimizer param group for the
    # named setting (lr or momentum). A list or tuple input must match the number of param
    # groups, otherwise a ValueError is raised naming the setting; a single value is
    # replicated for every group. Tensors are cloned so the returned entries never alias the
    # input. Args: name (for error text), optimizer, param.
    """Return correctly formatted lr/momentum for each param group."""

    def _copy(_param):
        # [FUNCTION DOC] HELPER: Return a clone of a tensor, or the value unchanged if it is
        # not a tensor.
        return _param.clone() if isinstance(_param, Tensor) else _param

    if isinstance(param, (list, tuple)):
        if len(param) != len(optimizer.param_groups):
            raise ValueError(
                f"{name} must have the same length as optimizer.param_groups. "
                f"{name} has {len(param)} values, param_groups has {len(optimizer.param_groups)}."
            )
    else:
        param = [param] * len(optimizer.param_groups)

    return list(map(_copy, param))


def _param_groups_val_list(optimizer: Optimizer, key: str) -> list[Any]:
    # [FUNCTION DOC] HELPER: Return a list holding the value stored under the key in each
    # param group, cloning tensor values so callers cannot corrupt the optimizer through
    # aliasing. Raises KeyError if any group lacks the key.
    """Create a list containing group[key] for each optimizer param_group.
    Prevents aliasing when group[key] could be a Tensor.
    Raises a KeyError when group[key] does not exist.
    """
    return [
        group[key].clone() if isinstance(group[key], Tensor) else group[key]
        for group in optimizer.param_groups
    ]


def _update_param_group_val(
    param_group: dict[str, Any], key: str, val: float | Tensor
) -> None:
    # [FUNCTION DOC] HELPER: Store a value under the key of one param group. If the existing
    # entry is a tensor it is filled in place with the scalar form of the value (preserving
    # identity, e.g. for captured graphs); otherwise the entry is replaced. Raises KeyError
    # if the key is missing. Mutates the param group.
    """Set param_group[key] to val without aliasing or assignment when they're
    both tensors. Raises a KeyError if param_group[key] does not exist.
    """
    if isinstance(param_group[key], Tensor):
        param_group[key].fill_(_to_scalar(val))
    else:
        param_group[key] = val


class LRScheduler:
    # [CLASS DOC] Base class of all lr schedulers. Wraps one optimizer and, on each step,
    # recomputes the lr of every param group and writes it back into the optimizer.
    # State: optimizer, base_lrs (initial lrs read from the initial_lr entry of the groups),
    # last_epoch (index of last step; -1 before initialization), _step_count, _last_lr
    # (copies of the most recent lrs). Class flags _get_lr_called_within_step and
    # _is_initial are toggled temporarily during a step.
    # Subclasses must implement get_lr and may provide a closed-form variant used when an
    # explicit epoch is passed to step. Subclasses that do not call this constructor
    # (SequentialLR, ChainedScheduler, ReduceLROnPlateau) manage their own state.
    r"""Base class for all learning rate schedulers.

    Subclasses implement :meth:`get_lr` and optionally override :meth:`step` to
    define scheduling behavior.

    Args:
        optimizer (Optimizer): The optimizer this scheduler will adjust the
            learning rates of.
        last_epoch (int): Index of the last epoch seen by the scheduler. Use
            ``-1`` (default) to initialize the scheduler. Only use a non-default
            value when restoring this scheduler from a saved checkpoint.

    .. warning::
        Initializing a scheduler overwrites its optimizer's
        ``param_group["lr"]``\s. When restoring a checkpoint, initialize the
        scheduler **before** calling your optimizer's
        :meth:`~torch.optim.Optimizer.load_state_dict` to avoid overwriting the
        loaded learning rates.
    """

    _get_lr_called_within_step: bool = False
    _is_initial: bool = False

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Attach the scheduler to an optimizer and perform the initial step.
        # Raises TypeError if the optimizer is not an Optimizer instance. When last_epoch is
        # -1, records each group's current lr as initial_lr (kept if already present);
        # otherwise every group must already contain initial_lr or a KeyError with advice is
        # raised.
        # Side effects: wraps the optimizer's step method once (marker attribute on the
        # wrapper) so that calling the optimizer is tracked in an attribute on the
        # optimizer, used to warn about wrong call order; then calls the initial step which
        # advances last_epoch and overwrites the groups' lr values.
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("initial_lr", initial_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        f"param 'initial_lr' is not specified in param_groups[{i}] when resuming scheduler with last_epoch >= 0.\n"
                        "This typically happens when:\n"
                        "1. You're trying to resume training from a checkpoint but haven't properly loaded the optimizer state\n"
                        "2. You're using last_epoch >= 0 for a fresh training run (not recommended)"
                    )
        self.base_lrs: list[float | Tensor] = _param_groups_val_list(
            optimizer, "initial_lr"
        )
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def patch_track_step_called(opt: Optimizer):
            # [FUNCTION DOC] HELPER: Make sure the optimizer's step method is wrapped so
            # each call sets a flag on the optimizer (used to detect scheduler stepping
            # before the optimizer). If already wrapped, returns the existing method;
            # otherwise replaces the optimizer's step attribute with the wrapper and returns
            # nothing (the return value is unused by the caller). Mutates the optimizer.
            if hasattr(opt.step, "_wrapped_by_lr_sched"):
                # we've already patched
                return opt.step

            def wrap_step(step_fn):
                # [FUNCTION DOC] HELPER: Build a wrapper around a bound optimizer step
                # method. Holds only a weak reference to the optimizer to avoid a reference
                # cycle, and tags the wrapper with a marker attribute. Returns the wrapper.
                opt_ref = ref(self.optimizer)
                func = step_fn.__func__

                @wraps(func)
                def wrapper(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Call the original step function on the (weakly
                    # referenced) optimizer after setting its called flag to true; forwards
                    # all arguments and returns the original result. Fails with an attribute
                    # error if the optimizer has been garbage collected.
                    opt = opt_ref()
                    opt._opt_called = True  # type: ignore[union-attr]
                    return func.__get__(opt, opt.__class__)(*args, **kwargs)

                wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
                return wrapper

            opt.step = wrap_step(opt.step)  # type: ignore[method-assign]

        patch_track_step_called(self.optimizer)
        self._initial_step()

    def _initial_step(self) -> None:
        # [FUNCTION DOC] HELPER: Reset the step counter to zero and call step once with the
        # initial-mode flag set, so subclasses can avoid applying a decay on the very first
        # step.
        """Initialize step counts and perform a step."""
        self._step_count = 0
        with _initial_mode(self):
            self.step()

    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a shallow dictionary of every instance attribute except the
        # optimizer. No side effects.
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore scheduler state by updating the instance attributes from
        # the given dictionary (typically one produced by state_dict). Mutates self; does
        # not copy or validate the input.
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Return the lrs most recently written to the optimizer (one per
        # param group), as stored copies that do not alias the optimizer. Raises an
        # attribute error if no step has run yet.
        r"""Get the most recent learning rates computed by this scheduler.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates with entries
            for each of the optimizer's
            :attr:`~torch.optim.Optimizer.param_groups`, with the same types as
            their ``group["lr"]``\s.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        # We always update self._last_lr with _param_groups_val_list, so it's a
        # .clone() of the group["lr"]s. If we didn't do this, the user could
        # corrupt their learning rates by modifying the outputs in place.
        return self._last_lr

    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Abstract hook: compute the next lr for each param group. The base
        # implementation always raises NotImplementedError. Subclasses return a list with
        # one float or tensor per group.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> None:
        # [FUNCTION DOC] Advance the scheduler by one step (or jump to an explicit epoch)
        # and write the new lrs to the optimizer.
        # On the first counted call, warns (UserWarning) if the optimizer step was
        # overridden after scheduler creation or if the optimizer has not been stepped yet.
        # Increments the step count; if epoch is given, emits a deprecation warning. Then
        # delegates to the lr update. Args: epoch (optional int, deprecated). Returns
        # nothing.
        """Step the scheduler.

        Args:
            epoch (int, optional):
                .. deprecated:: 1.4
                    If provided, sets :attr:`last_epoch` to ``epoch`` and uses
                    :meth:`_get_closed_form_lr` if it is available. This is not
                    universally supported. Use :meth:`step` without arguments
                    instead.

        .. note::
            Call this method after calling the optimizer's
            :meth:`~torch.optim.Optimizer.step`.
        """
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                    stacklevel=2,
                )

        self._step_count += 1
        if epoch is not None:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning, stacklevel=2)
        self._update_lr(epoch)

    def _update_lr(self, epoch: int | None = None) -> None:
        # [FUNCTION DOC] HELPER: Compute new lrs and apply them. With no epoch, increments
        # last_epoch and calls get_lr; with an epoch, sets last_epoch to it and uses the
        # closed-form method when the subclass defines one, else get_lr. get_lr runs with
        # the called-within-step flag set to suppress its warning.
        # Writes each value into the matching param group (in place for tensor lrs; the
        # number of values must equal the number of groups, enforced by strict zipping) and
        # refreshes _last_lr with copies.
        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = cast(list[float | Tensor], self._get_closed_form_lr())
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values, strict=True):
            _update_param_group_val(param_group, "lr", lr)

        self._last_lr: list[float | Tensor] = _param_groups_val_list(
            self.optimizer, "lr"
        )


def _warn_get_lr_called_within_step(lr_scheduler: LRScheduler) -> None:
    # [FUNCTION DOC] HELPER: Emit a UserWarning (pointing at the caller) if a scheduler's
    # get_lr is called by user code rather than from within a step, advising use of
    # get_last_lr instead.
    if not lr_scheduler._get_lr_called_within_step:
        warnings.warn(
            "To get the last learning rate computed by the scheduler, "
            "please use `get_last_lr()`.",
            UserWarning,
            stacklevel=2,
        )


# Including _LRScheduler for backwards compatibility
# Subclass instead of assign because we want __name__ of _LRScheduler to be _LRScheduler (assigning would make it LRScheduler).
class _LRScheduler(LRScheduler):
    # [CLASS DOC] HELPER: Backwards-compatibility alias subclass of LRScheduler. Kept as a
    # real subclass so its name stays _LRScheduler. Adds no behavior.
    pass


class _enable_get_lr_call:
    # [CLASS DOC] HELPER: Context manager that marks a scheduler as inside a step by setting
    # its get-lr-called-within-step flag on entry and clearing it on exit (even on error).
    # Used to silence the direct-call warning.
    def __init__(self, o: LRScheduler) -> None:
        # [FUNCTION DOC] HELPER: Store the scheduler to be flagged.
        self.o = o

    def __enter__(self) -> Self:
        # [FUNCTION DOC] HELPER: Set the flag on the scheduler to true and return this
        # context manager.
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback) -> None:
        # [FUNCTION DOC] HELPER: Clear the flag on the scheduler; exceptions are not
        # suppressed.
        self.o._get_lr_called_within_step = False


class _initial_mode:
    # [CLASS DOC] HELPER: Context manager that marks a scheduler as performing its initial
    # step by setting the is-initial flag on entry and clearing it on exit. Schedulers
    # consult the flag to avoid applying a change during construction.
    def __init__(self, o: LRScheduler) -> None:
        # [FUNCTION DOC] HELPER: Store the scheduler to be flagged.
        self.o = o

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Set the is-initial flag on the scheduler to true; returns
        # nothing.
        self.o._is_initial = True

    def __exit__(self, type, value, traceback):
        # [FUNCTION DOC] HELPER: Clear the is-initial flag; exceptions are not suppressed.
        self.o._is_initial = False


class LambdaLR(LRScheduler):
    # [CLASS DOC] Scheduler that sets each group's lr to its initial lr times a user
    # function of the epoch (one function for all groups or one per group). Holds lr_lambdas
    # (list of callables, one per group).
    # KNOWN ISSUE: the class docstring summary says it sets the initial learning rate, but
    #     get_lr multiplies base_lrs by the lambda output; it does not set initial_lr.
    """Sets the initial learning rate.

    The learning rate of each parameter group is set to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer has two groups.
        >>> num_epochs = 100
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95**epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
        >>>
        >>> # Alternatively, you can use a single lambda function for all groups.
        >>> scheduler = LambdaLR(opt, lr_lambda=lambda epoch: epoch // 30)
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LambdaLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store the lambdas: a single callable is replicated for all groups;
        # a list or tuple must match the number of param groups, otherwise a ValueError is
        # raised. Then runs the base constructor (which takes the initial step). Note the
        # optimizer attribute is assigned before the base class validates it. Args:
        # optimizer, lr_lambda, last_epoch.
        self.optimizer = optimizer

        self.lr_lambdas: list[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    @override
    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the scheduler state without the optimizer, with lr_lambdas
        # replaced by a list: plain functions and lambdas are stored as None, while callable
        # objects are stored as a copy of their attribute dictionary.
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state. Removes lr_lambdas from the given dictionary,
        # updates the instance attributes with the rest, then puts the entry back into the
        # caller's dictionary (temporary mutation to avoid side effects). For every non-None
        # saved entry, updates the attribute dictionary of the corresponding existing lambda
        # object.
        """Load the scheduler's state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Returns base_lrs of each group multiplied by the
        # matching lambda evaluated at last_epoch. Warns if called directly by user code.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the :attr:`base_lrs` by the outputs of the :attr:`lr_lambdas` at
        :attr:`last_epoch`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        return [
            base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs, strict=True)
        ]


class MultiplicativeLR(LRScheduler):
    # [CLASS DOC] Scheduler that multiplies each group's current lr by a factor given by a
    # user function of the epoch. Holds lr_lambdas (one callable per group). Chainable since
    # it scales the current lr rather than base_lrs.
    """Multiply the learning rate of each parameter group by the factor given in the specified function.

    When last_epoch=-1, set initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiplicativeLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store the lambdas (single callable replicated, or a list or tuple
        # matching the param group count else ValueError); every entry must be callable else
        # TypeError is raised. Then runs the base constructor. Args: optimizer, lr_lambda,
        # last_epoch.
        self.optimizer = optimizer

        self.lr_lambdas: list[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        for lr_lambda in self.lr_lambdas:
            if not callable(lr_lambda):
                raise TypeError(
                    f"lr_lambda should be a function, but got {type(lr_lambda).__name__}"
                )
        super().__init__(optimizer, last_epoch)

    @override
    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the scheduler state without the optimizer; lr_lambdas
        # becomes a list where functions and lambdas are None and callable objects are
        # stored as a copy of their attribute dictionary.
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state: temporarily removes lr_lambdas from the given
        # dictionary, updates attributes, restores the entry in the caller's dictionary, and
        # updates the attribute dictionary of each existing lambda object from its non-None
        # saved entry.
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Unless in the initial step, multiplies each group's
        # current lr by its lambda evaluated at last_epoch; during the initial step returns
        # the current lrs unchanged. Warns if called directly by user code.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the current ``group["lr"]``\s in each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by the outputs of the
        :attr:`lr_lambdas` at :attr:`last_epoch`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if not self._is_initial:
            return [
                group["lr"] * lmbda(self.last_epoch)
                for lmbda, group in zip(
                    self.lr_lambdas, self.optimizer.param_groups, strict=True
                )
            ]
        else:
            return _param_groups_val_list(self.optimizer, "lr")


class StepLR(LRScheduler):
    # [CLASS DOC] Scheduler that multiplies the lr of each group by gamma every step_size
    # epochs. Holds step_size and gamma. Chainable.
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/StepLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store step_size and gamma, then run the base constructor. Args:
        # optimizer, step_size (period), gamma (decay factor, default 0.1), last_epoch.
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Returns the current lrs unchanged unless last_epoch is a
        # nonzero multiple of step_size, in which case each is multiplied by gamma.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        If the current epoch is a non-zero multiple of :attr:`step_size`, we
        scale the current ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals base lr times gamma raised to the integer quotient of last_epoch by
        # step_size.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(LRScheduler):
    # [CLASS DOC] Scheduler that multiplies the lr by gamma when the epoch reaches a
    # milestone. Holds milestones as a counter (repeated milestones compound) and gamma.
    # Chainable.
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/MultiStepLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iterable[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store milestones as a multiset counter and gamma, then run the base
        # constructor. Args: optimizer, milestones (iterable of epoch indices), gamma,
        # last_epoch.
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. If last_epoch is not a milestone returns the current lrs
        # unchanged; otherwise multiplies each by gamma raised to the number of times that
        # epoch appears in the milestones.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        If the current epoch is in :attr:`milestones`, decays the
        ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.

        .. note::
            If the current epoch appears in :attr:`milestones` ``n`` times, we
            scale by :attr:`gamma` to the power of ``n``
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return _param_groups_val_list(self.optimizer, "lr")
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals base lr times gamma raised to the count of milestones not greater than
        # last_epoch.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class ConstantLR(LRScheduler):
    # [CLASS DOC] Scheduler that multiplies the lr by a constant factor until total_iters
    # steps have elapsed, then restores it. Holds factor and total_iters. Chainable.
    """Multiply the learning rate of each parameter group by a small constant factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # ...
        >>> # lr = 0.05    if epoch >= 40
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ConstantLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate that factor is between 0 and 1 (ValueError otherwise),
        # store factor and total_iters, then run the base constructor. Args: optimizer,
        # factor (default one third), total_iters (default 5), last_epoch.
        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. At epoch 0 multiplies the current lrs by factor; at
        # epoch equal to total_iters multiplies by the reciprocal of factor (fails with a
        # division error if factor is 0); at all other epochs returns the current lrs
        # unchanged.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        When :attr:`last_epoch` is 0, this method scales the ``group["lr"]``\s
        in each of the optimizer's :attr:`~torch.optim.Optimizer.param_groups`
        by :attr:`factor`. Once :attr:`total_iters` is reached, it undoes this,
        scaling by ``1 / factor``.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch != self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals base lr times factor before total_iters, and times 1 from total_iters
        # onward.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    # [CLASS DOC] Scheduler that linearly interpolates the lr multiplier from start_factor
    # to end_factor over total_iters steps, then holds it. Holds start_factor, end_factor,
    # total_iters. Chainable.
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.003687  if epoch == 0
        >>> # lr = 0.004875  if epoch == 1
        >>> # lr = 0.006062  if epoch == 2
        >>> # lr = 0.00725   if epoch == 3
        >>> # ...
        >>> # lr = 0.05      if epoch >= 40
        >>> scheduler = LinearLR(optimizer, start_factor=0.05, total_iters=40)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/LinearLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate start_factor (greater than 0 and at most 1) and end_factor
        # (between 0 and 1), raising ValueError otherwise; store the three parameters and
        # run the base constructor. Args: optimizer, start_factor, end_factor, total_iters,
        # last_epoch.
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. At epoch 0 multiplies current lrs by start_factor.
        # During the initial step or after total_iters returns the current lrs unchanged.
        # Otherwise multiplies each current lr by the ratio of the new to the previous
        # multiplier, so the cumulative factor moves linearly.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` such that successive steps
        interpolate linearly from :attr:`start_factor` up to :attr:`end_factor`
        across :attr:`total_iters` steps.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals base lr times a factor that moves linearly from start_factor to end_factor,
        # capped at total_iters.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]


class ExponentialLR(LRScheduler):
    # [CLASS DOC] Scheduler that multiplies the lr of each group by gamma every epoch. Holds
    # gamma. Chainable.
    """Decays the learning rate of each parameter group by gamma every epoch.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ExponentialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store gamma then run the base constructor. Args: optimizer, gamma,
        # last_epoch.
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Returns the current lrs unchanged during the initial
        # step (so loading a checkpoint does not decay once more); otherwise multiplies each
        # current lr by gamma.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Multiplies the current ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` by :attr:`gamma`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        # when loading from a checkpoint, we don't want _initial_step (called from the constructor)
        # to update the lr one more step ahead of itself.
        if self._is_initial:
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals base lr times gamma raised to last_epoch.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class SequentialLR(LRScheduler):
    # [CLASS DOC] Scheduler that runs a list of schedulers one after another, switching at
    # epoch milestones. Holds _schedulers, _milestones (one fewer than schedulers) and
    # mirrors last_epoch and _last_lr of the active scheduler.
    # Does not call the base constructor; it initializes its own state and drives the
    # wrapped schedulers directly.
    """Contains a list of schedulers expected to be called sequentially during the optimization process.

    Specifically, the schedulers will be called according to the milestone points, which should provide exact
    intervals by which each scheduler should be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.005     if epoch == 0
        >>> # lr = 0.005     if epoch == 1
        >>> # lr = 0.005     if epoch == 2
        >>> # ...
        >>> # lr = 0.05      if epoch == 20
        >>> # lr = 0.045     if epoch == 21
        >>> # lr = 0.0405    if epoch == 22
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(
        ...     optimizer,
        ...     schedulers=[scheduler1, scheduler2],
        ...     milestones=[20],
        ... )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/SequentialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[LRScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate and set up the sequence. Raises ValueError for an empty
        # scheduler list, for a ReduceLROnPlateau member, for a scheduler on a different
        # optimizer, or if milestones are not exactly one fewer than schedulers; raises
        # TypeError if a scheduler has no optimizer attribute.
        # Side effects: resets every group's lr to its initial_lr, undoes the initial step
        # that each wrapped scheduler already performed (decrementing last_epoch
        # recursively), then re-runs the initial step of only the scheduler active at epoch
        # 0 and records its last lrs. Args: optimizer, schedulers, milestones, last_epoch.
        # KNOWN ISSUE: last_epoch is set to the argument plus one, yet the initial scheduler
        #     is always chosen at epoch 0, so passing a non-default last_epoch is not
        #     honored.
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler."
            )

        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
                    "requires additional kwargs to be specified when calling `step`, "
                    f"but got one at index {scheduler_idx} in the given schedulers sequence."
                )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(schedulers)} and the "
                f"number of milestones to be equal to {len(milestones)}"
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        # Reset learning rates back to initial values
        for group in self.optimizer.param_groups:
            _update_param_group_val(group, "lr", group["initial_lr"])

        # "Undo" the step performed by other schedulers
        self.recursive_undo()

        # Perform the initial step for only the scheduler meant to run at step 0.
        idx = bisect_right(self._milestones, 0)
        self._schedulers[idx]._initial_step()

        self._last_lr = schedulers[idx].get_last_lr()

    def recursive_undo(self, sched=None) -> None:
        # [FUNCTION DOC] Recursively walk the given scheduler (default: this one) and, for
        # each leaf scheduler that has a last_epoch, decrement it by one; nested containers
        # are descended through their _schedulers. Mutates the schedulers. Public name,
        # though effectively internal.
        """
        Recursively undo any step performed by the initialization of
        schedulers.
        """
        scheds = self if sched is None else sched

        if hasattr(scheds, "_schedulers"):
            for s in scheds._schedulers:
                self.recursive_undo(s)
        elif hasattr(scheds, "last_epoch"):
            scheds.last_epoch -= 1

    def step(self) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Advance by one epoch. Picks the active scheduler by locating the
        # epoch among the milestones. If the epoch equals a milestone it restarts that
        # scheduler by calling its lr update with epoch 0; otherwise calls its normal step.
        # Copies the active scheduler's last lrs into _last_lr. Takes no epoch argument.
        """Perform a step."""
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler._update_lr(0)
        else:
            scheduler.step()

        self._last_lr = scheduler.get_last_lr()

    @override
    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the state without the optimizer, with _schedulers replaced
        # by a list of each wrapped scheduler's own state dictionary.
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            # pyrefly: ignore [unsupported-operation]
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state: temporarily removes _schedulers from the given
        # dictionary, updates attributes, restores the entry in the caller's dictionary,
        # then loads each saved state into the matching wrapped scheduler.
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class PolynomialLR(LRScheduler):
    # [CLASS DOC] Scheduler that decays the lr polynomially to zero over total_iters steps.
    # Holds total_iters and power. Chainable.
    # KNOWN ISSUE: the class docstring omits the last_epoch argument that the constructor
    #     accepts.
    """Decays the learning rate of each parameter group using a polynomial function in the given total_iters.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.0490   if epoch == 0
        >>> # lr = 0.0481   if epoch == 1
        >>> # lr = 0.0472   if epoch == 2
        >>> # ...
        >>> # lr = 0.0      if epoch >= 50
        >>> scheduler = PolynomialLR(optimizer, total_iters=50, power=0.9)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/PolynomialLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store total_iters and power, then run the base constructor. Args:
        # optimizer, total_iters (default 5), power (default 1.0), last_epoch.
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. During the initial step or when last_epoch exceeds
        # total_iters returns the current lrs unchanged; otherwise scales each by the ratio
        # of the new to the previous polynomial decay value.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` such that the learning rates
        follow

        .. math::
            \texttt{base\_lr} \cdot \left(1 - \frac{\texttt{last\_epoch}}
            {\texttt{total\_iters}} \right)^\texttt{power}

        Returns the current learning rates unchanged after :attr:`total_iters`
        is reached.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals base lr times one minus the progress fraction (capped at 1), raised to
        # power.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            (
                base_lr
                * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters)
                ** self.power
            )
            for base_lr in self.base_lrs
        ]


class CosineAnnealingLR(LRScheduler):
    # [CLASS DOC] Scheduler implementing cosine annealing without restarts between base lr
    # and eta_min over T_max steps, in a recursive form. Holds T_max and eta_min. Chainable.
    r"""
    Set the learning rate of each parameter group using a cosine annealing schedule.

    The learning rate is updated recursively using:

    .. math::
        \eta_{t+1} = \eta_{\min} + (\eta_t - \eta_{\min}) \cdot
        \frac{1 + \cos\left(\frac{(T_{cur}+1) \pi}{T_{max}}\right)}
            {1 + \cos\left(\frac{T_{cur} \pi}{T_{max}}\right)}

    This implements a recursive approximation of the closed-form schedule proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_:

    .. math::
        \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(
            1 + \cos\left(\frac{T_{cur} \pi}{T_{max}}\right) \right)

    where:

    - :math:`\eta_t` is the learning rate at step :math:`t`
    - :math:`T_{cur}` is the number of epochs since the last restart
    - :math:`T_{max}` is the maximum number of epochs in a cycle

    Note:
        Although SGDR includes periodic restarts, this implementation performs cosine annealing
        **without restarts**, so :math:`T_{cur} = t` and increases monotonically with each call
        to :meth:`step`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of the last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Example:
        >>> # xdoctest: +SKIP
        >>> num_epochs = 100
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/CosineAnnealingLR.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Store T_max and eta_min then run the base constructor. Args:
        # optimizer, T_max, eta_min (default 0), last_epoch.
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Branches: initial step returns current lrs; a first
        # counted step with a positive last_epoch (resuming) uses the closed-form value; at
        # epochs where the cosine wraps past a half period it uses an additive correction;
        # otherwise scales the distance from eta_min by the ratio of consecutive cosine
        # terms.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Scales the ``group["lr"]``\s in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` such that their learning
        rates approximate

        .. math::
            \texttt{eta\_min} + \frac{1}{2} (\texttt{base\_lr} -
            \texttt{eta\_min}) \left(1 + \cos\left(\pi \cdot
            \frac{\texttt{last\_epoch}}{\texttt{T\_max}}\right) \right)

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        if self._is_initial:
            return _param_groups_val_list(self.optimizer, "lr")
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(
                    self.base_lrs, self.optimizer.param_groups, strict=True
                )
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(
                    self.base_lrs, self.optimizer.param_groups, strict=True
                )
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] HELPER: Closed-form variant used when step receives an explicit
        # epoch; computed from base_lrs and last_epoch, independent of current lr values.
        # Equals eta_min plus half the distance from base lr to eta_min times one plus the
        # cosine of pi times last_epoch over T_max.
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.

        Uses :attr:`base_lrs` to compute learning rates. This method is called
        when an epoch is passed to :meth:`step`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.
        """
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]


class ChainedScheduler(LRScheduler):
    # [CLASS DOC] Scheduler that calls the step method of several wrapped schedulers in one
    # call, so their effects compose. Holds _schedulers and _last_lr.
    # Does not call the base constructor: base_lrs and last_epoch are not set on it.
    """Chains a list of learning rate schedulers.

    Takes in a sequence of chainable learning rate schedulers and calls their
    step() functions consecutively in just one call to step().

    Args:
        schedulers (sequence): sequence of chained schedulers.
        optimizer (Optimizer, optional): Wrapped optimizer. Default: None.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.005      if epoch == 0
        >>> # lr = 0.00450    if epoch == 1
        >>> # lr = 0.00405    if epoch == 2
        >>> # ...
        >>> # lr = 0.000675   if epoch == 19
        >>> # lr = 0.006078   if epoch == 20
        >>> # lr = 0.005470   if epoch == 21
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=20)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/ChainedScheduler.png
    """

    def __init__(
        self, schedulers: Sequence[LRScheduler], optimizer: Optimizer | None = None
    ) -> None:
        # [FUNCTION DOC] Validate and store the schedulers. Raises ValueError for an empty
        # sequence, a ReduceLROnPlateau member, or a scheduler on a different optimizer;
        # raises TypeError if a scheduler lacks an optimizer attribute. The optimizer
        # defaults to the first scheduler's optimizer. Initializes _last_lr from the current
        # lrs of the last scheduler's optimizer. Args: schedulers, optimizer (optional).
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler to be chained, but got no scheduler."
            )

        optimizer = optimizer or schedulers[0].optimizer
        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
                    "requires additional kwargs to be specified when calling `step`, "
                    f"but got one at index {scheduler_idx} in the given schedulers sequence."
                )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )
        self._schedulers = schedulers
        self.optimizer = optimizer
        self._last_lr = _param_groups_val_list(self._schedulers[-1].optimizer, "lr")

    def step(self) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Call step on every wrapped scheduler in order, then refresh
        # _last_lr from the optimizer of the last scheduler. Takes no epoch argument.
        """Perform a step."""
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = _param_groups_val_list(self._schedulers[-1].optimizer, "lr")

    @override
    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the state without the optimizer, with _schedulers replaced
        # by a list of each wrapped scheduler's state dictionary.
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            # pyrefly: ignore [unsupported-operation]
            state_dict["_schedulers"][idx] = s.state_dict()

        return state_dict

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state: temporarily removes _schedulers from the given
        # dictionary, updates attributes, restores the entry in the caller's dictionary,
        # then loads each saved state into the matching wrapped scheduler.
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)


class ReduceLROnPlateau(LRScheduler):
    # [CLASS DOC] Metric-driven scheduler that reduces the lr by a factor when a monitored
    # metric stops improving for more than patience epochs, with an optional cooldown
    # afterwards and a lower bound min_lrs per group.
    # State: factor, min_lrs and default_min_lr, patience, cooldown and cooldown_counter,
    # eps, mode and threshold settings, best, mode_worse, num_bad_epochs, last_epoch,
    # _last_lr. Not chainable; does not call the base constructor. step requires the metric.
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): The number of allowed epochs with no improvement after
            which the learning rate will be reduced.
            For example, consider the case of having no patience (`patience = 0`).
            In the first epoch, a baseline is established and is always considered good as there's no previous baseline.
            In the second epoch, if the performance is worse than the baseline,
            we have what is considered an intolerable epoch.
            Since the count of intolerable epochs (1) is greater than the patience level (0),
            the learning rate is reduced at the end of this epoch.
            From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch
            if the performance is worse than the baseline. If the performance improves or remains the same,
            the learning rate is not adjusted.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`.
            In `rel` mode, the dynamic threshold is computed as:

            * best * (1 + threshold) if mode == 'max'
            * best * (1 - threshold) if mode == 'min'

            In `abs` mode, the dynamic threshold is computed as:

            * best + threshold if mode == 'max'
            * best - threshold if mode == 'min'

            Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, "min")
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>> # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)

    .. image:: ../scripts/lr_scheduler_images/ReduceLROnPlateau.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_lr: list[float] | float = 0,
        eps: float = 1e-8,
    ) -> None:
        # [FUNCTION DOC] Validate and store settings. Raises ValueError if factor is 1.0 or
        # more, or if a list of min_lr values does not match the number of param groups;
        # raises TypeError if optimizer is not an Optimizer. A scalar min_lr is replicated
        # per group and remembered as default_min_lr. Sets last_epoch to 0, snapshots the
        # lrs, configures mode comparison (may raise ValueError for unknown mode) and resets
        # counters.
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.default_min_lr = None
            self.min_lrs = list(min_lr)
        else:
            self.default_min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps
        self.last_epoch = 0
        self._last_lr = _param_groups_val_list(self.optimizer, "lr")
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self) -> None:
        # [FUNCTION DOC] HELPER: Set best to the worst possible value for the mode and zero
        # the cooldown and bad-epoch counters.
        """Reset num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: SupportsFloat, epoch=None) -> None:  # type: ignore[override]
        # [FUNCTION DOC] Record a new metric value and reduce the lr if needed. Args:
        # metrics (float-like, a zero-dim tensor is converted to float), epoch (optional,
        # deprecated; warns).
        # Updates best and the bad-epoch count by comparing the metric; during cooldown
        # decrements the cooldown counter and ignores bad epochs. When bad epochs exceed
        # patience, reduces the lr, starts cooldown and clears the count. Finally snapshots
        # the lrs into _last_lr. Sets last_epoch.
        """Perform a step."""
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning, stacklevel=2)
        self.last_epoch = epoch

        if self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = _param_groups_val_list(self.optimizer, "lr")

    def _reduce_lr(self, epoch) -> None:
        # [FUNCTION DOC] HELPER: Multiply each group's lr by factor, bounded below by that
        # group's min lr, applying the change only if it exceeds eps. Raises RuntimeError if
        # the number of param groups changed since construction and no scalar default min_lr
        # is available; otherwise the min_lrs list is rebuilt from the default. Mutates the
        # optimizer.
        # KNOWN ISSUE: the epoch argument is never used.
        if len(self.optimizer.param_groups) != len(self.min_lrs):
            if self.default_min_lr is None:
                raise RuntimeError(
                    "The number of param groups in the `optimizer` "
                    f"({len(self.optimizer.param_groups)}) differs "
                    f"from when `ReduceLROnPlateau` was initialized "
                    f"({len(self.min_lrs)}), usually due to a new "
                    "param group being added to the optimizer. Please "
                    "modify the `min_lrs` field to match the length "
                    "of the `optimizer` param groups."
                )
            else:
                self.min_lrs = [self.default_min_lr] * len(self.optimizer.param_groups)

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                _update_param_group_val(param_group, "lr", new_lr)

    @property
    def in_cooldown(self):
        # [FUNCTION DOC] True while the cooldown counter is positive, meaning bad epochs are
        # currently being ignored.
        return self.cooldown_counter > 0

    def _is_better(self, a, best):
        # [FUNCTION DOC] HELPER: Return whether the metric a is better than best under the
        # configured mode (min or max) and threshold mode (relative or absolute), using the
        # threshold as a margin. Pure comparison, no state change.
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode) -> None:
        # [FUNCTION DOC] HELPER: Validate and store comparison settings. Raises ValueError
        # for an unknown mode or threshold mode. Sets mode_worse to positive infinity for
        # min mode or negative infinity for max mode, and stores mode, threshold and
        # threshold_mode.
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        # the worse value for the chosen mode
        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state by updating attributes from the given dictionary,
        # then re-run the comparison setup using the restored settings (this resets
        # mode_worse; best is taken from the saved state).
        """Load the scheduler's state."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )


class CyclicLR(LRScheduler):
    # [CLASS DOC] Scheduler implementing the cyclical lr policy: cycles lr between base_lr
    # and max_lr per group, with triangular, triangular2 or exp_range scaling or a custom
    # scaling function, and optionally cycles momentum (or Adam beta1) inversely.
    # State: max_lrs, base_lrs, total_size and step_ratio (derived from step_size_up and
    # step_size_down), mode, gamma, scale_mode, _scale_fn_custom, _scale_fn_ref,
    # cycle_momentum, base_momentums, max_momentums, use_beta1. Should be stepped once per
    # batch.
    r"""Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).

    The policy cycles the learning rate between two boundaries with a constant frequency,
    as detailed in the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLR(
        ...     optimizer,
        ...     base_lr=0.01,
        ...     max_lr=0.1,
        ...     step_size_up=10,
        ... )
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/CyclicLR.png

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float | list[float],
        max_lr: float | list[float],
        step_size_up: int = 2000,
        step_size_down: int | None = None,
        mode: Literal["triangular", "triangular2", "exp_range"] = "triangular",
        gamma: float = 1.0,
        scale_fn: Callable[[float], float] | None = None,
        scale_mode: Literal["cycle", "iterations"] = "cycle",
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate and configure the cycle. Raises TypeError for a non-
        # Optimizer, ValueError for an invalid mode without a custom scale function, or when
        # cycle_momentum is on but the optimizer has neither momentum nor betas defaults.
        # Per-group arguments may be lists matching the group count (ValueError otherwise).
        # Side effects: when last_epoch is -1, writes base lr into each group's lr and, if
        # cycling momentum, writes max momentum into the group's momentum or beta1 and
        # stores per-group max and base momentum entries. Runs the base constructor, then
        # overwrites base_lrs with the given base lrs.
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        base_lrs = _format_param("base_lr", optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups, strict=True):
                _update_param_group_val(group, "lr", lr)

        self.max_lrs = _format_param("max_lr", optimizer, max_lr)

        # pyrefly: ignore [bad-assignment]
        step_size_up = float(step_size_up)
        step_size_down = (
            # pyrefly: ignore [bad-assignment]
            float(step_size_down) if step_size_down is not None else step_size_up
        )
        # pyrefly: ignore [unsupported-operation]
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        self.mode = mode
        self.gamma = gamma

        self._scale_fn_ref: Callable[[float], float]
        self._scale_fn_custom = scale_fn
        self.scale_mode = scale_mode
        self._init_scale_fn()

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if (
                "momentum" not in optimizer.defaults
                and "betas" not in optimizer.defaults
            ):
                raise ValueError(
                    "optimizer must support momentum or beta1 with `cycle_momentum` option enabled"
                )

            self.use_beta1 = "betas" in self.optimizer.defaults
            self.base_momentums = _format_param(
                "base_momentum", optimizer, base_momentum
            )
            self.max_momentums = _format_param("max_momentum", optimizer, max_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(
                    self.max_momentums,
                    self.base_momentums,
                    optimizer.param_groups,
                    strict=True,
                ):
                    if self.use_beta1:
                        group["betas"] = (m_momentum, *group["betas"][1:])
                    else:
                        group["momentum"] = m_momentum
                    group["max_momentum"] = m_momentum
                    group["base_momentum"] = b_momentum

        super().__init__(optimizer, last_epoch)
        self.base_lrs = base_lrs

    def _init_scale_fn(self) -> None:
        # [FUNCTION DOC] HELPER: Select the internal scale function for the configured mode
        # (and scale_mode: cycle for the triangular modes, iterations for exp_range). Does
        # nothing when a custom scale function was given. Called at construction and after
        # loading state.
        if self._scale_fn_custom is not None:
            return
        if self.mode == "triangular":
            self._scale_fn_ref = self._triangular_scale_fn
            self.scale_mode = "cycle"
        elif self.mode == "triangular2":
            self._scale_fn_ref = self._triangular2_scale_fn
            self.scale_mode = "cycle"
        elif self.mode == "exp_range":
            self._scale_fn_ref = partial(self._exp_range_scale_fn, self.gamma)
            self.scale_mode = "iterations"

    def scale_fn(self, x) -> float:
        # [FUNCTION DOC] Evaluate the scaling policy at x: uses the custom function if
        # given, otherwise the built-in one for the mode. Returns a float between 0 and 1.
        """Get the scaling policy."""
        if self._scale_fn_custom is not None:
            return self._scale_fn_custom(x)
        else:
            return self._scale_fn_ref(x)  # static method

    @staticmethod
    def _triangular_scale_fn(x: float) -> float:
        # [FUNCTION DOC] HELPER: Constant scaling of 1; the argument is ignored.
        return 1.0

    @staticmethod
    def _triangular2_scale_fn(x: float) -> float:
        # [FUNCTION DOC] HELPER: Return the amplitude scale for triangular2 mode: one over
        # two to the power of (cycle number minus one), halving each cycle.
        return 1 / (2.0 ** (x - 1))

    @staticmethod
    def _exp_range_scale_fn(gamma: float, x: float) -> float:
        # [FUNCTION DOC] HELPER: Return gamma raised to the iteration count, the amplitude
        # scale for exp_range mode.
        return gamma**x

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Computes the position within the current cycle from
        # last_epoch and returns for each group base lr plus the amplitude (max minus base)
        # times the position factor times the scale function value (evaluated at the cycle
        # number or at last_epoch depending on scale_mode).
        # Side effect: if momentum cycling is enabled, writes the computed inverse momentum
        # into each group (beta1 of betas, or momentum).
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Advances each ``group["lr"]`` in the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` along a cycle between the
        group's ``base_lr`` and ``max_lr`` using :meth:`scale_fn`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.

        .. note::
            This method treats :attr:`last_epoch` as the index of the previous
            batch.

        .. note::
            When :attr:`cycle_momentum` is ``True``, this method has a side
            effect of updating the optimizer's momentum.
        """
        _warn_get_lr_called_within_step(self)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1.0 + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs, strict=True):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == "cycle":
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(
                self.base_momentums, self.max_momentums, strict=True
            ):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == "cycle":
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(
                        self.last_epoch
                    )
                momentums.append(momentum)
            for param_group, momentum in zip(
                self.optimizer.param_groups, momentums, strict=True
            ):
                if self.use_beta1:
                    param_group["betas"] = (momentum, *param_group["betas"][1:])
                else:
                    param_group["momentum"] = momentum

        return lrs

    @override
    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return the base-class state, dropping the unpicklable internal
        # scale function reference and replacing the custom scale function entry with None,
        # or a copy of its attribute dictionary if it is a callable object (plain functions
        # and lambdas are not saved).
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in ``self.__dict__`` which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """
        state = super().state_dict()
        # We are dropping the `_scale_fn_ref` attribute because it is a
        # `weakref.WeakMethod` and can't be pickled.
        state.pop("_scale_fn_ref", None)
        fn = state.pop("_scale_fn_custom")
        state["_scale_fn_custom"] = None
        if fn is not None and not isinstance(fn, types.FunctionType):
            # The _scale_fn_custom will only be saved if it is a callable object
            # and not if it is a function or lambda.
            state["_scale_fn_custom"] = fn.__dict__.copy()

        return state

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Restore state: removes the custom scale function entry, loads the
        # rest via the base class, updates the attribute dictionary of the existing custom
        # scale function from the saved entry if present, and rebuilds the internal scale
        # function.
        # KNOWN ISSUE: the removed entry is not put back into the caller's dictionary,
        #     unlike other schedulers, and it fails if a saved entry exists but no custom
        #     function is set.
        """Load the scheduler's state."""
        fn = state_dict.pop("_scale_fn_custom")
        super().load_state_dict(state_dict)
        if fn is not None:
            self._scale_fn_custom.__dict__.update(fn)
        self._init_scale_fn()


class CosineAnnealingWarmRestarts(LRScheduler):
    # [CLASS DOC] Scheduler implementing cosine annealing with warm restarts (SGDR). Holds
    # T_0 (first period), T_mult (period growth), T_i (current period), T_cur (position in
    # period) and eta_min. Step accepts fractional epochs.
    r"""Set the learning rate of each parameter group using a cosine annealing schedule.

    The :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations until the first restart.
        T_mult (int, optional): A factor by which :math:`T_{i}` increases after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        ...     optimizer, T_0=20
        ... )
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/CosineAnnealingWarmRestarts.png
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate and store parameters. Raises ValueError if T_0 is not a
        # positive integer, T_mult is not an integer of at least 1, or eta_min is not a
        # number. Sets T_i to T_0 and T_cur to last_epoch, then runs the base constructor.
        # Args: optimizer, T_0, T_mult, eta_min, last_epoch.
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Returns, for each base lr, eta_min plus half the
        # distance to eta_min times one plus the cosine of pi times T_cur over T_i.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Computes learning rates for the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` following:

        .. math::
            \texttt{eta\_min} + \frac{1}{2}(\texttt{base\_lr} -
            \texttt{eta\_min})\left(1 + \cos\left(\pi \cdot
            \frac{\texttt{T\_cur}}{\texttt{T\_i}}\right)\right)

        Where :attr:`T_cur` is the number of epochs since the last restart and
        :attr:`T_i` is the number of epochs between two restarts. Both
        :attr:`T_cur` and :attr:`T_i` are updated in :meth:`step`, and
        :attr:`T_i` becomes :attr:`T_mult` times larger after each restart.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.
        """
        _warn_get_lr_called_within_step(self)

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    @override
    def step(self, epoch=None) -> None:
        # [FUNCTION DOC] Advance the schedule and write new lrs to the optimizer. With no
        # epoch, moves T_cur forward by one, restarting (and multiplying T_i by T_mult) when
        # the period ends; at a negative last_epoch it starts from epoch 0. With an epoch
        # (may be fractional), recomputes T_cur and T_i directly from the epoch; raises
        # ValueError for a negative epoch. Sets last_epoch to the floor of the epoch,
        # applies get_lr to the groups and refreshes _last_lr.
        # KNOWN ISSUE: does not increment the step count nor run the base-class call-order
        #     warnings; the docstring example comment mentions scheduler(20) where step(20)
        #     is meant.
        """Step could be called after every batch update.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur % self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        with _enable_get_lr_call(self):
            for param_group, lr in zip(
                self.optimizer.param_groups, self.get_lr(), strict=True
            ):
                _update_param_group_val(param_group, "lr", lr)

        self._last_lr = _param_groups_val_list(self.optimizer, "lr")


class _SchedulePhase(TypedDict):
    # [CLASS DOC] HELPER: Typed dictionary describing one phase of a OneCycleLR schedule:
    # the end step and the names of the param group keys used for start and end lr and
    # momentum. Static typing aid only.
    end_step: float
    start_lr: str
    end_lr: str
    start_momentum: str
    end_momentum: str


class OneCycleLR(LRScheduler):
    # [CLASS DOC] Scheduler implementing the 1cycle policy: lr anneals from initial lr up to
    # max_lr and then down to a minimum, over total_steps, in two or three phases, with
    # optional inverse momentum cycling. Stepped once per batch; not chainable.
    # State: total_steps, _schedule_phases (list of phase descriptions), _anneal_func_type,
    # cycle_momentum, use_beta1. Per-group entries initial_lr, max_lr, min_lr (and
    # max_momentum, base_momentum) are written into the optimizer's param groups.
    r"""Sets the learning rate of each parameter group according to the 1cycle learning rate policy.

    The 1cycle policy anneals the learning rate from an initial learning rate to some maximum
    learning rate and then from that maximum learning rate to some minimum learning rate much
    lower than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
    claims that "unpublished work has shown even better results by using only two phases". To
    mimic the behaviour of the original paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to annihilate the
            learning rate according to 'final_div_factor' instead of modifying the second
            phase (the first two phases will be symmetrical about the step indicated by
            'pct_start').
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> # xdoctest: +SKIP
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(
        ...     optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10
        ... )
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         optimizer.step()
        >>>         scheduler.step()

    .. image:: ../scripts/lr_scheduler_images/OneCycleLR.png

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | list[float],
        total_steps: int | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        cycle_momentum: bool = True,
        base_momentum: float | list[float] = 0.85,
        max_momentum: float | list[float] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC] Validate and configure. Raises TypeError for a non-Optimizer;
        # ValueError if total_steps is not a positive integer, if neither total_steps nor
        # both epochs and steps_per_epoch are given, if epochs or steps_per_epoch is not a
        # positive integer, if pct_start is not a float between 0 and 1, if anneal_strategy
        # is unknown, or if momentum cycling is requested but the optimizer has no momentum
        # or betas defaults.
        # Builds the phase list (two phases, or three if three_phase). When last_epoch is
        # -1, writes initial_lr (max_lr over div_factor), max_lr and min_lr (initial over
        # final_div_factor) into each group and, if cycling momentum, sets momentum or beta1
        # to max momentum and stores momentum bounds. Then runs the base constructor.
        # KNOWN ISSUE: pct_start is validated after it is already used to build the phases,
        #     and an int pct_start is rejected only because of the float type check.
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError(
                    f"Expected positive integer total_steps, but got {total_steps}"
                )
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is not None:
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError(f"Expected positive integer epochs, but got {epochs}")
            if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
                raise ValueError(
                    f"Expected positive integer steps_per_epoch, but got {steps_per_epoch}"
                )
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError(
                "You must define either total_steps OR (epochs AND steps_per_epoch)"
            )

        self._schedule_phases: list[_SchedulePhase]
        if three_phase:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": float(2 * pct_start * self.total_steps) - 2,
                    "start_lr": "max_lr",
                    "end_lr": "initial_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "max_momentum",
                },
            ]
        else:
            self._schedule_phases = [
                {
                    "end_step": float(pct_start * self.total_steps) - 1,
                    "start_lr": "initial_lr",
                    "end_lr": "max_lr",
                    "start_momentum": "max_momentum",
                    "end_momentum": "base_momentum",
                },
                {
                    "end_step": self.total_steps - 1,
                    "start_lr": "max_lr",
                    "end_lr": "min_lr",
                    "start_momentum": "base_momentum",
                    "end_momentum": "max_momentum",
                },
            ]

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError(
                f"Expected float between 0 and 1 pct_start, but got {pct_start}"
            )

        # Validate anneal_strategy
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"anneal_strategy must be one of 'cos' or 'linear', instead got {anneal_strategy}"
            )
        else:
            self._anneal_func_type = anneal_strategy

        # Initialize learning rate variables
        max_lrs = _format_param("max_lr", self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group["initial_lr"] = max_lrs[idx] / div_factor
                group["max_lr"] = max_lrs[idx]
                group["min_lr"] = group["initial_lr"] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if (
                "momentum" not in self.optimizer.defaults
                and "betas" not in self.optimizer.defaults
            ):
                raise ValueError(
                    "optimizer must support momentum or beta1 with `cycle_momentum` option enabled"
                )
            self.use_beta1 = "betas" in self.optimizer.defaults
            max_momentums = _format_param("max_momentum", optimizer, max_momentum)
            base_momentums = _format_param("base_momentum", optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(
                    max_momentums, base_momentums, optimizer.param_groups, strict=True
                ):
                    if self.use_beta1:
                        group["betas"] = (m_momentum, *group["betas"][1:])
                    else:
                        group["momentum"] = m_momentum
                    group["max_momentum"] = m_momentum
                    group["base_momentum"] = b_momentum

        super().__init__(optimizer, last_epoch)

    def _anneal_func(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Dispatch to the cosine or linear annealing function
        # according to the configured strategy; raises ValueError for an unknown type. Falls
        # back to an older instance attribute for backward compatibility with scheduler
        # objects restored without the type. Returns the annealed value.
        if hasattr(self, "_anneal_func_type"):
            if self._anneal_func_type == "cos":
                return self._annealing_cos(*args, **kwargs)
            elif self._anneal_func_type == "linear":
                return self._annealing_linear(*args, **kwargs)
            else:
                raise ValueError(f"Unknown _anneal_func_type: {self._anneal_func_type}")
        else:
            # For BC
            return self.anneal_func(*args, **kwargs)  # type: ignore[attr-defined]

    @staticmethod
    def _annealing_cos(start, end, pct):
        # [FUNCTION DOC] HELPER: Cosine-anneal from start to end as pct goes from 0 to 1;
        # returns a float.
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start, end, pct):
        # [FUNCTION DOC] HELPER: Linearly interpolate from start to end as pct goes from 0
        # to 1; returns a float.
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        return (end - start) * pct + start

    @override
    def get_lr(self) -> list[float | Tensor]:
        # [FUNCTION DOC] Computes the next lr for every param group; the result is assigned
        # to the optimizer by step. Raises ValueError if last_epoch exceeds total_steps. For
        # each group finds the phase containing the current step, computes the progress
        # fraction within the phase and anneals lr (and momentum, if cycling) between the
        # phase's start and end values read from the group.
        # Side effect: when momentum cycling is on, writes the computed momentum (or beta1)
        # into each group.
        # KNOWN ISSUE: the last valid step index is total_steps minus one, but only larger
        #     values are rejected, so total_steps itself yields an extrapolated lr; a phase
        #     whose end step equals its start step would divide by zero.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Finds the appropriate :attr:`_schedule_phases` entry for the current
        step and interpolates between its ``start_lr`` and ``end_lr`` using
        :meth:`_anneal_func`.

        Returns:
            list[float | Tensor]: A :class:`list` of learning rates for each of
            the optimizer's :attr:`~torch.optim.Optimizer.param_groups` with the
            same types as their current ``group["lr"]``\s.

        .. note::
            If you're trying to inspect the most recent learning rate, use
            :meth:`get_last_lr()` instead.

        .. note::
            The returned :class:`~torch.Tensor`\s are copies, and never alias
            the optimizer's ``group["lr"]``\s.

        .. note::
            When :attr:`cycle_momentum` is ``True``, this method has a side
            effect of updating the optimizer's momentum.
        """
        _warn_get_lr_called_within_step(self)

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. The specified number of total steps is {self.total_steps}"
            )

        for group in self.optimizer.param_groups:
            start_step = 0.0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_lr = self._anneal_func(
                        group[phase["start_lr"]], group[phase["end_lr"]], pct
                    )
                    if self.cycle_momentum:
                        computed_momentum = self._anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break
                start_step = phase["end_step"]

            lrs.append(computed_lr)  # type: ignore[possibly-undefined]
            if self.cycle_momentum:
                if self.use_beta1:
                    group["betas"] = (computed_momentum, *group["betas"][1:])  # type: ignore[possibly-undefined]
                else:
                    group["momentum"] = computed_momentum  # type: ignore[possibly-undefined]

        return lrs
