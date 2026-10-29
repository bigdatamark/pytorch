# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Exponential distribution (exponential family) parameterized by a
# positive rate, with reparameterized sampling and closed-form moments, cdf,
# icdf and entropy. No import-time side effects.
# Exports: Exponential.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Exponential"]


class Exponential(ExponentialFamily):
    # [CLASS DOC]
    # Exponential distribution on nonnegative reals. State: rate tensor (positive,
    # broadcast). Supports reparameterized sampling; the mean carrier measure is
    # the constant zero.
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: the reciprocal of rate.
        return self.rate.reciprocal()

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: zeros shaped like rate.
        return torch.zeros_like(self.rate)

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC] Standard deviation: the reciprocal of rate.
        return self.rate.reciprocal()

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance: rate raised to the power minus two.
        return self.rate.pow(-2)

    def __init__(
        self,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores rate (number or tensor, converted to a tensor); batch shape is empty for a Python number, else the rate shape. Base class validation may raise ValueError if rate is not positive.
        (self.rate,) = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, _Number) else self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Exponential with rate expanded to batch_shape, preserving the validation setting; self is unchanged.
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws samples of shape sample_shape plus batch shape: standard exponential noise (consumes random state) divided by rate, so gradients flow to rate.
        shape = self._extended_shape(sample_shape)
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value: log of rate minus rate times value. Validates value if validation is on (ValueError if outside support).
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value):
        # [FUNCTION DOC] Cumulative distribution: one minus exp of minus rate times value. Validates value if validation is on.
        if self._validate_args:
            self._validate_sample(value)
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        # [FUNCTION DOC] Inverse cdf: minus log1p of minus value, divided by rate. No input validation.
        return -torch.log1p(-value) / self.rate

    def entropy(self):
        # [FUNCTION DOC] Entropy: one minus log of rate.
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self) -> tuple[Tensor]:
        # [FUNCTION DOC] HELPER: Natural parameters, a one-element tuple holding negative rate.
        return (-self.rate,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        # [FUNCTION DOC] HELPER: Log normalizer at natural parameter x: minus log of minus x.
        return -torch.log(-x)
