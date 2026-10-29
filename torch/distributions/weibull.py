# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the two-parameter Weibull distribution as a TransformedDistribution
# of a unit Exponential through a power then scale transform. Exports Weibull.
# No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.gumbel import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all


__all__ = ["Weibull"]


class Weibull(TransformedDistribution):
    # [CLASS DOC]
    # Weibull distribution with positive scale and concentration (shape),
    # broadcast together. State: scale, concentration, concentration_reciprocal
    # and the inherited base_dist/transforms. Samples are a unit exponential
    # raised to the reciprocal concentration then multiplied by scale. Support
    # is the positive reals. Provides mean, mode, variance and entropy.
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive

    def __init__(
        self,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Broadcasts scale and concentration, stores the reciprocal of the
        # concentration, builds a unit Exponential base and the transform list
        # (power by reciprocal concentration, then affine scaling), and
        # delegates to TransformedDistribution. Validation may raise ValueError.
        self.scale, self.concentration = broadcast_all(scale, concentration)
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Exponential(
            torch.ones_like(self.scale), validate_args=validate_args
        )
        transforms = [
            PowerTransform(exponent=self.concentration_reciprocal),
            AffineTransform(loc=0, scale=self.scale),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC]
        # Returns a new Weibull with parameters, reciprocal, base distribution
        # and transforms rebuilt for batch_shape; validation setting is copied.
        new = self._get_checked_instance(Weibull, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.concentration_reciprocal = new.concentration.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        transforms = [
            PowerTransform(exponent=new.concentration_reciprocal),
            AffineTransform(loc=0, scale=new.scale),
        ]
        super(Weibull, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Scale times the gamma function of one plus the reciprocal concentration.
        return self.scale * torch.exp(torch.lgamma(1 + self.concentration_reciprocal))

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC]
        # Scale times ((k-1)/k) to the power 1/k, with k the concentration.
        # KNOWN ISSUE: for concentration below one the base is negative, giving
        # NaN, whereas the true Weibull mode is zero when concentration <= 1.
        return (
            self.scale
            * ((self.concentration - 1) / self.concentration)
            ** self.concentration.reciprocal()
        )

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC]
        # Squared scale times the difference between gamma of one plus twice
        # the reciprocal concentration and the squared gamma of one plus the
        # reciprocal concentration (computed via log-gamma and exponentials).
        return self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 + 2 * self.concentration_reciprocal))
            - torch.exp(2 * torch.lgamma(1 + self.concentration_reciprocal))
        )

    def entropy(self):
        # [FUNCTION DOC]
        # Differential entropy: Euler constant times (1 - 1/k) plus log of
        # scale/k plus one, with k the concentration.
        return (
            euler_constant * (1 - self.concentration_reciprocal)
            + torch.log(self.scale * self.concentration_reciprocal)
            + 1
        )
