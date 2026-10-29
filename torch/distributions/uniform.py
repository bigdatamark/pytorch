# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Uniform distribution over the half-open interval [low, high).
# Exports Uniform. No import-time side effects.

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Uniform"]


class Uniform(Distribution):
    # [CLASS DOC]
    # Continuous uniform distribution parameterized by tensors low and high
    # (broadcast to a common shape; the batch shape is that shape, or empty if
    # both were plain numbers). Supports reparameterized sampling, cdf, icdf,
    # log_prob and entropy. Invariant: low is less than high.
    r"""
    Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.

    Example::

        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
        >>> # xdoctest: +SKIP
        tensor([ 2.3418])

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """

    has_rsample = True

    @property
    def arg_constraints(self):
        # [FUNCTION DOC]
        # Mapping of parameter names to constraints: low must be less than the
        # current high and high greater than the current low.
        # TODO allow (loc,scale) parameterization to allow independent constraints.
        return {
            "low": constraints.less_than(self.high),
            "high": constraints.greater_than(self.low),
        }

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Midpoint of low and high.
        return (self.high + self.low) / 2

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # The mode is undefined for a uniform distribution; returns NaN shaped
        # like high.
        return nan * self.high

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC]
        # Standard deviation: interval width divided by the square root of 12.
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Variance: squared interval width divided by 12.
        return (self.high - self.low).pow(2) / 12

    def __init__(
        self,
        low: Tensor | float,
        high: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Broadcasts low and high into tensors stored as self.low/self.high.
        # Batch shape is empty when both inputs are plain numbers, otherwise
        # the broadcast shape. Validates arguments if requested (may raise
        # ValueError from the base class).
        self.low, self.high = broadcast_all(low, high)

        if isinstance(low, _Number) and isinstance(high, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Uniform with low and high expanded to batch_shape;
        # validation setting is copied from self.
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(Uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        # [FUNCTION DOC]
        # Interval constraint between low and high (continuous, scalar events).
        return constraints.interval(self.low, self.high)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC]
        # Draws standard uniform random numbers (consuming RNG state) of the
        # extended shape and maps them linearly onto [low, high). Differentiable
        # with respect to low and high.
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Log density: negative log of the width inside the interval, negative
        # infinity outside (the lower bound is inclusive, upper exclusive).
        # Optionally validates value against the support.
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, value):
        # [FUNCTION DOC]
        # Linear cumulative probability of value, clamped to the range zero to
        # one. Optionally validates value.
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def icdf(self, value):
        # [FUNCTION DOC]
        # Maps probability value linearly into the interval. Does not validate
        # or clamp value.
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self):
        # [FUNCTION DOC]
        # Differential entropy: log of the interval width.
        return torch.log(self.high - self.low)
