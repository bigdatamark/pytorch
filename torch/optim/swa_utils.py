# mypy: allow-untyped-defs
# [MODULE DOC] Utilities for Stochastic Weight Averaging (SWA) and Exponential Moving Average
# (EMA) of model weights. Provides AveragedModel (a wrapper keeping an averaged copy of a
# model), update_bn (recompute batch-norm statistics), the SWALR scheduler, and factory
# functions returning per-parameter or multi-parameter averaging callables. No import-time
# side effects beyond defining names.
r"""Implementation for Stochastic Weight Averaging implementation."""

from __future__ import annotations

import itertools
import math
import warnings
from copy import deepcopy
from typing import Any, cast, Literal, TYPE_CHECKING
from typing_extensions import override

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import _format_param, LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .optimizer import Optimizer


__all__ = [
    "AveragedModel",
    "update_bn",
    "SWALR",
    "get_ema_multi_avg_fn",
    "get_swa_multi_avg_fn",
    "get_ema_avg_fn",
    "get_swa_avg_fn",
]

from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


PARAM_LIST = tuple[Tensor, ...] | list[Tensor]


def get_ema_multi_avg_fn(decay=0.999):
    # [FUNCTION DOC] Builds an in-place EMA update callable for lists of parameters.
    # Args: decay, weight of the old average, must lie in [0, 1]; otherwise raises an error.
    # Returns a function taking (ema list, current list, ignored count) that moves each EMA
    # tensor toward the current one by (1 - decay): via a fused foreach interpolation for
    # float/complex dtypes, else a per-tensor copy of the weighted sum (integer dtypes).
    """Get the function applying exponential moving average (EMA) across multiple params.

    The EMA is computed as:

    .. math::
        W_0^{\\text{EMA}} = W_0^{\\text{model}}

    .. math::
        W_{t+1}^{\\text{EMA}} = \\text{decay} \\times W_t^{\\text{EMA}} + (1 - \\text{decay}) \\times W_{t+1}^{\\text{model}}

    where :math:`W_t^{\\text{EMA}}` is the EMA parameter at step :math:`t`,
    :math:`W_t^{\\text{model}}` is the model parameter at step :math:`t`,
    and :math:`\\text{decay}` is the decay rate (default: 0.999).

    Args:
        decay (float): Decay rate for EMA. Must be in the range [0, 1]. Default: 0.999

    Returns:
        Callable: A function that updates EMA parameters given current model parameters
    """

    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(
        ema_param_list: PARAM_LIST, current_param_list: PARAM_LIST, _
    ) -> None:
        # [FUNCTION DOC] HELPER: Closure doing the in-place EMA update of the listed tensors
        # using the captured decay; the dtype of the first tensor selects the foreach path.
        # foreach lerp only handles float and complex
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(
            ema_param_list[0]
        ):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    return ema_update


def get_swa_multi_avg_fn():
    # [FUNCTION DOC] Returns an in-place running-mean update callable for lists of parameters.
    # The callable takes (averaged list, current list, number averaged so far) and moves each
    # averaged tensor toward the current one by 1 / (num_averaged + 1).
    """Get the function applying stochastic weight average (SWA) across multiple params."""

    @torch.no_grad()
    def swa_update(
        averaged_param_list: PARAM_LIST,
        current_param_list: PARAM_LIST,
        num_averaged: Tensor | int,
    ) -> None:
        # [FUNCTION DOC] HELPER: Closure applying the running-mean update in place. Float and
        # complex tensors use a foreach interpolation; other dtypes add the scaled difference,
        # dividing by a tensor count or scaling by a reciprocal for a plain integer count.
        # foreach lerp only handles float and complex
        if torch.is_floating_point(averaged_param_list[0]) or torch.is_complex(
            averaged_param_list[0]
        ):
            torch._foreach_lerp_(
                averaged_param_list,
                current_param_list,
                cast(float, 1 / (num_averaged + 1)),
            )
        else:
            diffs = torch._foreach_sub(current_param_list, averaged_param_list)
            if isinstance(num_averaged, Tensor):
                torch._foreach_addcdiv_(
                    averaged_param_list,
                    diffs,
                    [num_averaged + 1] * len(averaged_param_list),
                )
            else:
                torch._foreach_add_(
                    averaged_param_list, diffs, alpha=1.0 / (num_averaged + 1)
                )

    return swa_update


def get_ema_avg_fn(decay=0.999):
    # [FUNCTION DOC] Builds a single-tensor EMA callable (returns a new value, no mutation).
    # Args: decay, must lie in [0, 1]; otherwise raises an error.
    # KNOWN ISSUE: the docstring says "across multiple params" but this works on one tensor.
    """Get the function applying exponential moving average (EMA) across multiple params.

    The EMA is computed as:

    .. math::
        W_0^{\\text{EMA}} = W_0^{\\text{model}}

    .. math::
        W_{t+1}^{\\text{EMA}} = \\text{decay} \\times W_t^{\\text{EMA}} + (1 - \\text{decay}) \\times W_{t+1}^{\\text{model}}

    where :math:`W_t^{\\text{EMA}}` is the EMA parameter at step :math:`t`,
    :math:`W_t^{\\text{model}}` is the model parameter at step :math:`t`,
    and :math:`\\text{decay}` is the decay rate (default: 0.999).

    Args:
        decay (float): Decay rate for EMA. Must be in the range [0, 1]. Default: 0.999

    Returns:
        Callable: A function that updates EMA parameters given current model parameters
    """

    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged):
        # [FUNCTION DOC] HELPER: Returns decay times the EMA tensor plus (1 - decay) times the
        # current tensor; the count argument is ignored.
        return decay * ema_param + (1 - decay) * current_param

    return ema_update


def get_swa_avg_fn():
    # [FUNCTION DOC] Returns a single-tensor running-mean callable (no mutation).
    """Get the function applying stochastic weight average (SWA) across a single param."""

    @torch.no_grad()
    def swa_update(
        averaged_param: Tensor, current_param: Tensor, num_averaged: Tensor | int
    ):
        # [FUNCTION DOC] HELPER: Returns the averaged tensor plus the difference to the current
        # tensor divided by (num_averaged + 1).
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)

    return swa_update


class AveragedModel(Module):
    # [CLASS DOC] Module wrapper holding a deep copy of a model whose parameters (and optionally
    # buffers) are maintained as a running average of a source model, for SWA or EMA.
    # State: module (the copy), n_averaged (long buffer counting updates), avg_fn or
    # multi_avg_fn (optional custom averaging; at most one), use_buffers flag. Forward delegates
    # to the copy. The first update copies weights; later updates apply the averaging function.
    r"""Implements averaged model for Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    Exponential Moving Average is a variation of `Polyak averaging`_,
    but using exponential weights instead of equal weights across iterations.

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA/EMA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        multi_avg_fn (function, optional): the averaging function used to update
            parameters inplace; the function must take in the current values of the
            :class:`AveragedModel` parameters as a list, the current values of :attr:`model`
            parameters as a list, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with the `avg_fn` or `multi_avg_fn` parameters.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights (SWA).

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_model = torch.optim.swa_utils.AveragedModel(model,
        >>>             torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)

    .. note::
        When using SWA/EMA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` and `multi_avg_fn` are not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    .. _Polyak averaging:
        https://paperswithcode.com/method/polyak-averaging
    """

    n_averaged: Tensor

    def __init__(
        self,
        model: Module,
        device: int | torch.device | None = None,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, Tensor | int], None]
        | None = None,
        use_buffers=False,
    ) -> None:
        # [FUNCTION DOC] Deep-copies the model (moved to device if given), registers the
        # n_averaged counter buffer (starting at zero) and stores the averaging options.
        # Raises an error if both avg_fn and multi_avg_fn are given.
        super().__init__()
        if avg_fn is not None and multi_avg_fn is not None:
            raise AssertionError(
                "Only one of avg_fn and multi_avg_fn should be provided"
            )
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the wrapped averaged copy on the given arguments and returns its
        # output.
        """Forward pass."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: Module) -> None:
        # [FUNCTION DOC] Folds the source model's current weights into the running average.
        # Pairs parameters (plus buffers if use_buffers) positionally with the source model.
        # On the first call (n_averaged is zero) copies the source values in. Afterwards:
        # with a single-tensor avg_fn, applies it per tensor; otherwise groups tensors by device
        # and dtype and applies multi_avg_fn, or the default SWA running mean (foreach kernel
        # when the device supports it, else per tensor). Without use_buffers, buffers are copied
        # straight from the source. Increments n_averaged. Mutates tensors in place.
        # KNOWN ISSUE: pairing of parameters uses non-strict zip, so a structural mismatch
        # between the models is silently truncated rather than reported.
        """Update model parameters."""
        self_param = (
            # pyrefly: ignore [bad-argument-type]
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        model_param = (
            # pyrefly: ignore [bad-argument-type]
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )
        self_param_detached: list[Tensor | None] = []
        model_param_detached: list[Tensor | None] = []
        copy_param = bool(self.n_averaged == 0)
        for p_averaged, p_model in zip(self_param, model_param, strict=False):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if copy_param:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped_tensors = _group_tensors_by_device_and_dtype(
                    [self_param_detached, model_param_detached]
                )
                for (device, _), (
                    [self_params, model_params],
                    _,
                ) in grouped_tensors.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(
                            self_params,  # type: ignore[arg-type]
                            model_params,  # type: ignore[arg-type]
                            self.n_averaged.to(device),
                        )
                    elif (
                        device is not None
                        and device.type in _get_foreach_kernels_supported_devices()
                    ):
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(
                            self_params, model_params, self.n_averaged.to(device)
                        )
                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        for p_averaged, p_model in zip(  # type: ignore[assignment]
                            self_params, model_params, strict=True
                        ):
                            # pyrefly: ignore [missing-attribute]
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
            else:
                for p_averaged, p_model in zip(  # type: ignore[assignment]
                    self_param_detached, model_param_detached, strict=True
                ):
                    # pyrefly: ignore [missing-attribute]
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    # pyrefly: ignore [missing-attribute]
                    p_averaged.detach().copy_(
                        # pyrefly: ignore [missing-attribute, bad-argument-type]
                        self.avg_fn(p_averaged.detach(), p_model, n_averaged)
                    )

        if not self.use_buffers:
            # If not apply running averages to the buffers,
            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(
                self.module.buffers(), model.buffers(), strict=True
            ):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1


@torch.no_grad()
def update_bn(
    loader: Iterable[Any],
    model: Module,
    device: int | torch.device | None = None,
) -> None:
    # [FUNCTION DOC] Recomputes BatchNorm running statistics with one pass over the loader.
    # Args: loader (iterable of tensors or lists/tuples whose first item is the input),
    # model, device (optional destination for inputs). Resets running stats of every batch-norm
    # module, sets momentum to None (cumulative average), switches to train mode, runs the
    # forward pass on each batch, then restores momenta and the previous training mode.
    # Returns early, doing nothing, if the model has no batch-norm layers. No gradients.
    r"""Update BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta:
        module.momentum = None

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta:
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(LRScheduler):
    # [CLASS DOC] Learning-rate scheduler that anneals each param group's lr toward a fixed
    # swa_lr over anneal_epochs epochs, then holds it there. State: anneal_epochs, the chosen
    # anneal function (cosine or linear, stored by name and as a bound method), and a per-group
    # swa_lr entry written into the optimizer's param groups. State dict excludes the optimizer
    # and the anneal function.
    r"""Anneals the learning rate in each parameter group to a fixed value.

    This learning rate scheduler is meant to be used with Stochastic Weight
    Averaging (SWA) method (see `torch.optim.swa_utils.AveragedModel`).

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lrs (float or list): the learning rate value for all param groups
            together or separately for each group.
        annealing_epochs (int): number of epochs in the annealing phase
            (default: 10)
        annealing_strategy (str): "cos" or "linear"; specifies the annealing
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: -1)

    The :class:`SWALR` scheduler can be used together with other
    schedulers to switch to a constant learning rate late in the training
    as in the example below.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, optimizer, model = ...
        >>> lr_lambda = lambda epoch: 0.9
        >>> scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        >>>        lr_lambda=lr_lambda)
        >>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
        >>>        anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
        >>> swa_start = 160
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    """

    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs=10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch=-1,
    ) -> None:
        # [FUNCTION DOC] Stores swa_lr (a number or per-group list) into each param group,
        # selects the anneal function and anneal_epochs, then initializes the base scheduler.
        # Raises an error for an unknown strategy or an anneal_epochs that is not a
        # non-negative integer. Side effect: adds a swa_lr key to every optimizer param group.
        swa_lrs = _format_param("swa_lr", optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups, strict=True):
            group["swa_lr"] = swa_lr
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must be one of 'cos' or 'linear', "
                f"instead got {anneal_strategy}"
            )
        self._set_anneal_func(anneal_strategy)
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(
                f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}"
            )
        self.anneal_epochs = anneal_epochs
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _linear_anneal(t):
        # [FUNCTION DOC] HELPER: Identity mapping of progress t in [0, 1] to blend weight.
        return t

    @staticmethod
    def _cosine_anneal(t):
        # [FUNCTION DOC] HELPER: Maps progress t in [0, 1] to a blend weight along a half cosine
        # curve (0 at start, 1 at end).
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        # [FUNCTION DOC] HELPER: Inverts one annealing step to recover the lr the schedule
        # started from, given current lr, target swa_lr and blend weight alpha. When alpha is 1
        # (division would be by zero) returns swa_lr.
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    @override
    def get_lr(self):
        # [FUNCTION DOC] Computes the next lr for every param group. Warns if called outside a
        # scheduler step. Derives the previous and current blend weights from the step count
        # (clamped progress over anneal_epochs; with zero epochs jumps straight to the target),
        # reconstructs each group's starting lr from the previous weight, then returns the
        # mix of swa_lr and starting lr by the current weight. Returns a list, one per group.
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.

        Uses :attr:`anneal_func` to interpolate between each group's
        ``group["lr"]`` and ``group["swa_lr"]`` over :attr:`anneal_epochs`
        epochs. Once :attr:`anneal_epochs` is reached, keeps the learning rate
        fixed at ``group["swa_lr"]``.

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
        # `_get_lr_called_within_step` is only available `_enable_get_lr_call`,
        # so we ignore the type error here. See `LRScheduler.step()` for more details.
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        # Set in `LRScheduler._initial_step()`
        step = self._step_count - 1
        if self.anneal_epochs == 0:
            step = max(1, step)
        # pyrefly: ignore [no-matching-overload]
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [
            self._get_initial_lr(group["lr"], group["swa_lr"], prev_alpha)
            for group in self.optimizer.param_groups
        ]
        # pyrefly: ignore [no-matching-overload]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [
            group["swa_lr"] * alpha + lr * (1 - alpha)
            for group, lr in zip(self.optimizer.param_groups, prev_lrs, strict=True)
        ]

    def _set_anneal_func(self, anneal_strategy: Literal["cos", "linear"]) -> None:
        # [FUNCTION DOC] HELPER: Records the strategy name and binds anneal_func to the cosine
        # function for cos, else the linear function.
        self._anneal_strategy = anneal_strategy
        if anneal_strategy == "cos":
            self.anneal_func = self._cosine_anneal
        else:
            self.anneal_func = self._linear_anneal

    @override
    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a new mapping of all instance attributes except the optimizer
        # and the anneal function.
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer or anneal_func.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "anneal_func")
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC] Merges the given mapping into the instance attributes, then rebinds the
        # anneal function from the restored strategy name. Mutates the scheduler in place.
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self._set_anneal_func(self._anneal_strategy)
