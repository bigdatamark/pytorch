# mypy: allow-untyped-defs
# [MODULE DOC] Batch normalization layers: BatchNorm1d/2d/3d, their lazy variants that infer
# num_features from the first input, and SyncBatchNorm which synchronizes statistics across
# distributed processes. Private bases _NormBase (shared parameters/buffers, state-dict version
# handling), _BatchNorm (forward via the functional batch_norm) and _LazyNormBase (deferred
# parameter creation) hold the shared logic. Exports the names in __all__. No import-time side effects.
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter, UninitializedBuffer, UninitializedParameter

from ._functions import SyncBatchNorm as sync_batch_norm
from .lazy import LazyModuleMixin
from .module import Module


__all__ = [
    "BatchNorm1d",
    "LazyBatchNorm1d",
    "BatchNorm2d",
    "LazyBatchNorm2d",
    "BatchNorm3d",
    "LazyBatchNorm3d",
    "SyncBatchNorm",
]


class _NormBase(Module):
    # [CLASS DOC] HELPER: Shared base of batch and instance norm layers.
    # State: num_features, eps, momentum (None means cumulative average), affine, track_running_stats.
    # Parameters weight and bias exist only when affine (bias may be disabled separately); buffers
    # running_mean, running_var and num_batches_tracked exist only when track_running_stats, otherwise
    # they are registered as None. Class version is 2 (version 2 added num_batches_tracked).
    # Subclasses must implement the input dimension check.
    """Common base of _InstanceNorm and _BatchNorm."""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float | None
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        *,
        bias: bool = True,
    ) -> None:
        # [FUNCTION DOC] Create the affine parameters and running-statistics buffers, then reset them.
        # Args: num_features, eps, momentum, affine, track_running_stats, device, dtype, bias
        # (keyword-only; whether the affine shift exists). num_batches_tracked is a long scalar that
        # ignores the requested dtype. Side effect: parameters initialized to weight 1, bias 0, mean 0, var 1.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_var", torch.ones(num_features, **factory_kwargs)
            )
            self.running_mean: Tensor | None
            self.running_var: Tensor | None
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    # pyrefly: ignore [bad-argument-type]
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: Tensor | None
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        # [FUNCTION DOC] Reset running mean to zeros, running variance to ones and the batch counter to
        # zero in place; does nothing when running statistics are not tracked.
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        # [FUNCTION DOC] Reset running statistics, then (if affine) set weight to ones and bias to zeros
        # in place when it exists.
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def _check_input_dim(self, input):
        # [FUNCTION DOC] HELPER: Abstract input validation hook; the base version always raises
        # NotImplementedError and is overridden by subclasses to check input dimensionality.
        raise NotImplementedError

    def extra_repr(self):
        # [FUNCTION DOC] Return the repr suffix listing num_features, eps, momentum, affine, whether bias
        # exists, and track_running_stats.
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "bias={use_bias}, track_running_stats={track_running_stats}".format(
                **self.__dict__, use_bias=self.bias is not None
            )
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # [FUNCTION DOC] HELPER: Load hook for state dicts. For checkpoints saved before module version 2
        # that track running stats, inserts a missing num_batches_tracked entry (the current buffer, or a
        # zero long scalar when the buffer is absent or on the meta device) into the state dict (mutated in
        # place), then defers to the parent loader which fills missing/unexpected keys and error messages.
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = (
                    self.num_batches_tracked
                    if self.num_batches_tracked is not None
                    and self.num_batches_tracked.device != torch.device("meta")
                    else torch.tensor(0, dtype=torch.long)
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    # [CLASS DOC] HELPER: Base for the concrete batch norm layers. Adds the forward pass on top of
    # _NormBase; subclasses supply the input dimension check. No extra state.
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        *,
        bias: bool = True,
    ) -> None:
        # [FUNCTION DOC] HELPER: Forward construction arguments to the _NormBase constructor unchanged.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            **factory_kwargs,
            bias=bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Normalize the input per channel using batch statistics or running statistics.
        # Validates input dimensionality (subclass hook), then chooses the averaging factor: momentum, or
        # 1 / batches-seen for cumulative averaging when momentum is None (incrementing
        # num_batches_tracked in training mode). Batch statistics are used in training mode, or in eval
        # mode when running stats are absent. Running buffers are passed (and thus updated in training)
        # only when tracked. Returns a tensor of the input's shape. Raises ValueError from the check hook.
        # Side effects: increments num_batches_tracked and updates running buffers in training mode.
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class _LazyNormBase(LazyModuleMixin, _NormBase):
    # [CLASS DOC] HELPER: Mixin base providing lazy batch norm: num_features is unknown at
    # construction. weight, bias and running buffers start uninitialized and are materialized on the
    # first forward from the input's channel dimension, after which the layer turns into its
    # non-lazy class (via cls_to_become).
    weight: UninitializedParameter  # type: ignore[assignment]
    bias: UninitializedParameter  # type: ignore[assignment]

    def __init__(
        self,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
        *,
        bias=True,
    ) -> None:
        # [FUNCTION DOC] HELPER: Construct with num_features 0 and no real tensors, then record the
        # requested affine / track_running_stats settings and create uninitialized weight, bias and
        # running statistics placeholders (plus a zero batch counter). Args: eps, momentum, affine,
        # track_running_stats, device, dtype, bias (keyword-only).
        factory_kwargs = {"device": device, "dtype": dtype}
        # pyrefly: ignore [bad-argument-type]
        super().__init__(
            # affine, bias and track_running_stats are hardcoded to False to
            # avoid creating tensors that will soon be overwritten.
            0,
            eps,
            momentum,
            False,
            False,
            **factory_kwargs,
            bias=False,
        )
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # pyrefly: ignore [unexpected-keyword]
            self.weight = UninitializedParameter(**factory_kwargs)
            if bias:
                # pyrefly: ignore  # bad-argument-type
                self.bias = UninitializedParameter(**factory_kwargs)
        if self.track_running_stats:
            # pyrefly: ignore [unexpected-keyword]
            self.running_mean = UninitializedBuffer(**factory_kwargs)
            # pyrefly: ignore [unexpected-keyword]
            self.running_var = UninitializedBuffer(**factory_kwargs)
            self.num_batches_tracked = torch.tensor(
                0,
                dtype=torch.long,
                # pyrefly: ignore [bad-argument-type]
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            )

    def reset_parameters(self) -> None:
        # [FUNCTION DOC] Reset parameters only once everything has been materialized and num_features is
        # nonzero; otherwise does nothing.
        # pyrefly: ignore [bad-argument-type]
        if not self.has_uninitialized_params() and self.num_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        # [FUNCTION DOC] HELPER: Materialize lazy tensors from the first input. Sets num_features to the
        # size of dimension 1 of the input, materializes weight, bias and running buffers to that length
        # and resets parameters. Does nothing if nothing is uninitialized. Raises AssertionError if
        # weight or bias are not uninitialized parameters.
        # pyrefly: ignore [bad-argument-type]
        if self.has_uninitialized_params():
            self.num_features = input.shape[1]
            if self.affine:
                if not isinstance(self.weight, UninitializedParameter):
                    raise AssertionError(
                        "self.weight must be an UninitializedParameter"
                    )
                self.weight.materialize((self.num_features,))
                if self.bias is not None:
                    if not isinstance(self.bias, UninitializedParameter):
                        raise AssertionError(
                            "self.bias must be an UninitializedParameter"
                        )
                    self.bias.materialize((self.num_features,))
            if self.track_running_stats:
                self.running_mean.materialize(  # type:ignore[union-attr]
                    (self.num_features,)
                )
                self.running_var.materialize(  # type:ignore[union-attr]
                    (self.num_features,)
                )
            self.reset_parameters()


class BatchNorm1d(_BatchNorm):
    # [CLASS DOC] Batch normalization over 2D (N, C) or 3D (N, C, L) input, normalizing each channel
    # over the batch (and length) dimensions. Inherits all state and behavior from _BatchNorm; only the
    # accepted input dimensionality is defined here.
    r"""Applies Batch Normalization over a 2D or 3D input.

    Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input). By default, the
    elements of :math:`\gamma` are set to 1 and the elements of :math:`\beta` are set to 0.
    At train time in the forward pass, the variance is calculated via the biased estimator,
    equivalent to ``torch.var(input, correction=0)``. However, the value stored in the
    moving average of the variance is calculated via the unbiased estimator, equivalent to
    ``torch.var(input, correction=1)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: number of features or channels :math:`C` of the input
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`, where :math:`N` is the batch size,
          :math:`C` is the number of features or channels, and :math:`L` is the sequence length
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input is 2D or 3D.
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class LazyBatchNorm1d(_LazyNormBase, _BatchNorm):
    # [CLASS DOC] BatchNorm1d whose num_features is inferred from the first input (dimension 1).
    # After materialization its class becomes BatchNorm1d (cls_to_become).
    r"""A :class:`torch.nn.BatchNorm1d` module with lazy initialization.

    Lazy initialization based on the ``num_features`` argument of the :class:`BatchNorm1d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``
    """

    cls_to_become = BatchNorm1d  # type: ignore[assignment]

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input is 2D or 3D.
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    # [CLASS DOC] Batch normalization over 4D (N, C, H, W) input. Inherits all state and behavior from
    # _BatchNorm; only the accepted input dimensionality is defined here.
    r"""Applies Batch Normalization over a 4D input.

    4D is a mini-batch of 2D inputs
    with additional channel dimension. Method described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, correction=0)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased estimator, equivalent to
    ``torch.var(input, correction=1)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input is 4D.
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class LazyBatchNorm2d(_LazyNormBase, _BatchNorm):
    # [CLASS DOC] BatchNorm2d whose num_features is inferred from the first input (dimension 1).
    # After materialization its class becomes BatchNorm2d (cls_to_become).
    r"""A :class:`torch.nn.BatchNorm2d` module with lazy initialization.

    Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm2d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``
    """

    cls_to_become = BatchNorm2d  # type: ignore[assignment]

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input is 4D.
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")


class BatchNorm3d(_BatchNorm):
    # [CLASS DOC] Batch normalization over 5D (N, C, D, H, W) input. Inherits all state and behavior
    # from _BatchNorm; only the accepted input dimensionality is defined here.
    r"""Applies Batch Normalization over a 5D input.

    5D is a mini-batch of 3D inputs with additional channel dimension as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. At train time in the forward pass, the
    standard-deviation is calculated via the biased estimator, equivalent to
    ``torch.var(input, correction=0)``. However, the value stored in the moving average of the
    standard-deviation is calculated via the unbiased estimator, equivalent to
    ``torch.var(input, correction=1)``.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input is 5D.
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class LazyBatchNorm3d(_LazyNormBase, _BatchNorm):
    # [CLASS DOC] BatchNorm3d whose num_features is inferred from the first input (dimension 1).
    # After materialization its class becomes BatchNorm3d (cls_to_become).
    r"""A :class:`torch.nn.BatchNorm3d` module with lazy initialization.

    Lazy initialization is done for the ``num_features`` argument of the :class:`BatchNorm3d` that is inferred
    from the ``input.size(1)``.
    The attributes that will be lazily initialized are `weight`, `bias`,
    `running_mean` and `running_var`.

    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation
    on lazy modules and their limitations.

    Args:
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``
    """

    cls_to_become = BatchNorm3d  # type: ignore[assignment]

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input is 5D.
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")


class SyncBatchNorm(_BatchNorm):
    # [CLASS DOC] Batch normalization whose statistics are synchronized across all processes of a
    # distributed process group during training, for N-D inputs with at least 2 dimensions.
    # State: as _BatchNorm plus process_group (None means the world group). Falls back to ordinary
    # batch norm in eval mode, without an initialized distributed setup, or with a world size of 1.
    # Use convert_sync_batchnorm to convert an existing model.
    r"""Applies Batch Normalization over a N-Dimensional input.

    The N-D input is a mini-batch of [N-2]D inputs with additional channel dimension as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, correction=0)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.

    Currently :class:`SyncBatchNorm` only supports
    :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`affine` is ``True``). Default: ``True``

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    .. note::
        Synchronization of batchnorm statistics occurs only while training, i.e.
        synchronization is disabled when ``model.eval()`` is set or if
        ``self.training`` is otherwise ``False``.

    Examples::

        >>> # xdoctest: +SKIP
        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))
        >>> r1, r2 = ranks[:4], ranks[4:]
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
        >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Any | None = None,
        device=None,
        dtype=None,
        *,
        bias: bool = True,
    ) -> None:
        # [FUNCTION DOC] Build like _BatchNorm and store the process group.
        # Args: num_features, eps, momentum, affine, track_running_stats, process_group (optional),
        # device, dtype, bias (keyword-only).
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            **factory_kwargs,
            bias=bias,
        )
        self.process_group = process_group

    def _check_input_dim(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError unless the input has at least 2 dimensions.
        if input.dim() < 2:
            raise ValueError(f"expected at least 2D input (got {input.dim()}D input)")

    def _check_non_zero_input_channels(self, input) -> None:
        # [FUNCTION DOC] HELPER: Raise ValueError if the channel dimension (size of dimension 1) is zero.
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Normalize the input, synchronizing statistics across processes when needed.
        # Validates the input, computes the averaging factor like the regular batch norm (incrementing
        # num_batches_tracked in training; raises AssertionError if that buffer is None), and picks
        # batch statistics in training or when no running stats exist. Synchronization happens only in
        # training with torch.distributed initialized and a group world size above 1; then the input must
        # be on cuda, hpu, xpu or the privateuse1 device (else ValueError) and a custom autograd function
        # exchanges statistics. Otherwise the functional batch_norm is used. Returns same-shaped tensor.
        # KNOWN ISSUE: the device error message mentions GPU, XPU and privateuse1 but not hpu, which is
        # also accepted.
        """
        Runs the forward pass.
        """
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is None:
                raise AssertionError("num_batches_tracked must not be None")
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (
            bn_training
            and self.training
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if need_sync:
            # currently only GPU/PrivateUse1 input is supported
            if input.device.type not in [
                "cuda",
                "hpu",
                "xpu",
                torch._C._get_privateuse1_backend_name(),
            ]:
                raise ValueError(
                    "SyncBatchNorm expected input tensor to be on GPU or XPU or "
                    f"{torch._C._get_privateuse1_backend_name()}"
                )

            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            if not bn_training:
                raise AssertionError("bn_training must be True")
            return sync_batch_norm.apply(
                input,
                self.weight,
                self.bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,  # type: ignore[possibly-undefined]
                world_size,  # type: ignore[possibly-undefined]
            )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        # [FUNCTION DOC] Recursively replace every batch norm layer (instances of the private base class)
        # in a module tree with SyncBatchNorm.
        # Args: module, process_group (optional). Returns the converted module; if the root itself is a
        # batch norm layer a new SyncBatchNorm is returned. Converted layers share the original weight,
        # bias, running statistics and counter tensors, copy the training flag and any qconfig attribute.
        # Children are converted recursively and re-attached by name; the input reference is deleted locally.
        r"""Converts all :attr:`BatchNorm*D` layers in the model to :class:`torch.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> # xdoctest: +SKIP("distributed")
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
                bias=module.bias is not None,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output
