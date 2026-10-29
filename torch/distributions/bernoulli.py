# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Bernoulli distribution over {0, 1}, parameterized by either probs
# or logits. Exports Bernoulli via __all__. No import-time side effects.

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, Number


__all__ = ["Bernoulli"]


class Bernoulli(ExponentialFamily):
    # [CLASS DOC]
    # Bernoulli distribution with exactly one of probs or logits supplied; the
    # other is computed lazily on first access and cached. _param records which
    # one was given and defines the batch shape, dtype and device. Support is
    # the boolean set, enumerable, no rsample. Is an exponential family.
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
        validate_args (bool, optional): whether to validate arguments, None by default
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    _mean_carrier_measure = 0

    def __init__(
        self,
        probs: Tensor | Number | None = None,
        logits: Tensor | Number | None = None,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the supplied parameter (broadcast to a tensor) and sets _param
        # to it. Batch shape is empty for a plain number, otherwise the tensor
        # shape. Raises ValueError if both or neither of probs and logits are
        # given.
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
        else:
            if logits is None:
                raise AssertionError("logits is unexpectedly None")
            is_scalar = isinstance(logits, _Number)
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Bernoulli with whichever of probs/logits has already
        # been materialized expanded to batch_shape; reuses _instance if given.
        # Preserves the validate_args setting. Does not mutate self.
        new = self._get_checked_instance(Bernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Bernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: creates a new tensor of the same type and
        # device as _param, forwarding all arguments to its new method.
        return self._param.new(*args, **kwargs)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: returns probs.
        return self.probs

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: 1 where probs is at least one half, else 0, in probs dtype;
        # entries where probs is exactly one half are set to nan (mode undefined).
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Property: probs times one minus probs.
        return self.probs * (1 - self.probs)

    @lazy_property
    def logits(self) -> Tensor:
        # [FUNCTION DOC]
        # Lazy cached property: converts probs to binary logits when logits were
        # not supplied at construction.
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        # [FUNCTION DOC]
        # Lazy cached property: converts logits to binary probabilities when
        # probs were not supplied at construction.
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        # [FUNCTION DOC]
        # Property: shape of the stored parameter tensor.
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC]
        # Draws binary samples of shape sample_shape plus batch shape from probs
        # expanded to that shape, without gradient tracking. Consumes global
        # RNG state.
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Returns elementwise log probability of value: the negative binary
        # cross entropy with logits, after broadcasting logits and value.
        # Validates value when validation is enabled (raises ValueError).
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self):
        # [FUNCTION DOC]
        # Returns the entropy as the binary cross entropy with logits of probs
        # against itself, with no reduction (shape equals batch shape).
        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )

    def enumerate_support(self, expand=True):
        # [FUNCTION DOC]
        # Returns the values 0 and 1 as a tensor of the parameter dtype/device,
        # shaped with leading dimension two followed by singleton batch dims; if
        # expand is true, expanded to the full batch shape.
        values = torch.arange(2, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

    @property
    def _natural_params(self) -> tuple[Tensor]:
        # [FUNCTION DOC] HELPER: exponential-family natural parameter, a
        # one-element tuple holding the logit of probs.
        return (torch.logit(self.probs),)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        # [FUNCTION DOC] HELPER: log normalizer, log of one plus exp of x.
        return torch.log1p(torch.exp(x))
