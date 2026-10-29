# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the InverseGamma distribution as a TransformedDistribution: the
# reciprocal (power minus one) of a Gamma variable. No import-time side effects.
# Exports: InverseGamma.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform


__all__ = ["InverseGamma"]


class InverseGamma(TransformedDistribution):
    # [CLASS DOC]
    # Inverse gamma distribution on positive reals. State: a Gamma base
    # distribution (held by the transformed-distribution base class) and a power
    # transform with exponent minus one; concentration and rate are exposed from
    # the base. Supports reparameterized sampling.
    r"""
    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Gamma

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a Gamma base from concentration and rate (validating if requested, which may raise ValueError) and wraps it in a power transform with exponent minus one (a scalar of the rate dtype and device).
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        neg_one = -base_dist.rate.new_ones(())
        super().__init__(
            base_dist, PowerTransform(neg_one), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new InverseGamma with batch dims expanded to batch_shape via the parent class; self is unchanged.
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self) -> Tensor:
        # [FUNCTION DOC] Concentration of the underlying Gamma base distribution.
        return self.base_dist.concentration

    @property
    def rate(self) -> Tensor:
        # [FUNCTION DOC] Rate of the underlying Gamma base distribution.
        return self.base_dist.rate

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC]
        # Mean: rate over (concentration minus one) where concentration exceeds 1, positive infinity elsewhere.
        # KNOWN ISSUE: for concentration at most 1 the mean is nonexistent (divergent); returning infinity is a convention, not NaN.
        result = self.rate / (self.concentration - 1)
        return torch.where(self.concentration > 1, result, torch.inf)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: rate over (concentration plus one).
        return self.rate / (self.concentration + 1)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance: rate squared over (concentration minus one) squared times (concentration minus two) where concentration exceeds 2, positive infinity elsewhere (a divergence convention).
        result = self.rate.square() / (
            (self.concentration - 1).square() * (self.concentration - 2)
        )
        return torch.where(self.concentration > 2, result, torch.inf)

    def entropy(self):
        # [FUNCTION DOC] Differential entropy: concentration plus log rate plus log-gamma of concentration minus (1 plus concentration) times digamma of concentration.
        return (
            self.concentration
            + self.rate.log()
            + self.concentration.lgamma()
            - (1 + self.concentration) * self.concentration.digamma()
        )
