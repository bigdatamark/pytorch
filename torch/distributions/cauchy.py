# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Cauchy (Lorentz) distribution, a location-scale distribution with
# reparameterized sampling and closed-form cdf, icdf and entropy. Exports Cauchy
# via __all__. No import-time side effects.
import math

import torch
from torch import inf, nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Cauchy"]


class Cauchy(Distribution):
    # [CLASS DOC]
    # Cauchy distribution with tensor attributes loc (median/mode) and scale
    # (half width at half maximum, must be positive), broadcast to a common
    # shape. Support is the real line; supports rsample. Mean and variance are
    # undefined and reported as nan and infinity respectively.
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
        tensor([ 2.3214])

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores loc and scale after broadcasting them to a common shape. Batch
        # shape is empty when both inputs are plain numbers, otherwise the shape
        # of the broadcast loc. Delegates to Distribution init with validate_args.
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Cauchy with loc and scale expanded (as views) to
        # batch_shape; reuses _instance if given. Preserves the validate_args
        # setting of self. Does not mutate self.
        new = self._get_checked_instance(Cauchy, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Cauchy, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Property returning a tensor of nan with the extended shape (the mean of
        # a Cauchy distribution does not exist), matching loc dtype and device.
        return torch.full(
            self._extended_shape(), nan, dtype=self.loc.dtype, device=self.loc.device
        )

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Property returning loc, the mode of the distribution.
        return self.loc

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Property returning a tensor of positive infinity with the extended
        # shape, matching loc dtype and device (variance is unbounded).
        return torch.full(
            self._extended_shape(), inf, dtype=self.loc.dtype, device=self.loc.device
        )

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC]
        # Draws a reparameterized sample of shape sample_shape plus batch shape:
        # fills a standard Cauchy noise tensor in place via the Tensor cauchy_
        # method, then scales by scale and shifts by loc. Consumes global RNG
        # state; gradients flow to loc and scale.
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new(shape).cauchy_()
        return self.loc + eps * self.scale

    def log_prob(self, value):
        # [FUNCTION DOC]
        # Returns the elementwise log density of value (broadcast with the
        # parameters). Validates value against the support when validation is
        # enabled (raising ValueError on violation).
        if self._validate_args:
            self._validate_sample(value)
        return (
            -math.log(math.pi)
            - self.scale.log()
            - (((value - self.loc) / self.scale) ** 2).log1p()
        )

    def cdf(self, value):
        # [FUNCTION DOC]
        # Returns the cumulative distribution at value using the arctangent of
        # the standardized value, scaled to the unit interval. Validates value
        # when validation is enabled.
        if self._validate_args:
            self._validate_sample(value)
        return torch.atan((value - self.loc) / self.scale) / math.pi + 0.5

    def icdf(self, value):
        # [FUNCTION DOC]
        # Returns the quantile function (inverse cdf) at probability value using
        # the tangent transform, then scale and shift. No input validation.
        return torch.tan(math.pi * (value - 0.5)) * self.scale + self.loc

    def entropy(self):
        # [FUNCTION DOC]
        # Returns the differential entropy: log of four pi plus log of scale.
        # Shape follows scale.
        return math.log(4 * math.pi) + self.scale.log()
