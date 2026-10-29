# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the HalfNormal distribution, the absolute value of a zero-mean Normal
# variable, as a TransformedDistribution over Normal with an absolute value
# transform. No import-time side effects. Exports: HalfNormal.
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


__all__ = ["HalfNormal"]


class HalfNormal(TransformedDistribution):
    # [CLASS DOC]
    # Half-normal distribution on nonnegative reals. State: the base Normal
    # distribution (mean 0, given scale) held by the transformed-distribution
    # base class; scale is exposed from it. Supports reparameterized sampling.
    r"""
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # half-normal distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the full Normal distribution
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Normal

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a Normal base with mean 0 and the given scale (validation disabled on the base) and wraps it with an absolute value transform; validate_args applies to the half-normal itself and may raise ValueError for nonpositive scale.
        base_dist = Normal(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new HalfNormal with batch dims expanded to batch_shape via the parent class; self is unchanged.
        new = self._get_checked_instance(HalfNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        # [FUNCTION DOC] Scale of the underlying Normal base distribution.
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: scale times the square root of 2 over pi.
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: zeros shaped like scale.
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance: scale squared times one minus 2 over pi.
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value: the Normal log density plus log 2 for nonnegative values, negative infinity for negative ones. Validates value if validation is on (ValueError). Without validation, value must already support comparison with zero (a tensor).
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        # [FUNCTION DOC] Cumulative distribution: twice the Normal cdf minus one. Validates value if validation is on.
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        # [FUNCTION DOC] Inverse cdf of prob: the Normal inverse cdf at (prob plus one) over two.
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        # [FUNCTION DOC] Entropy: the Normal base entropy minus log 2.
        return self.base_dist.entropy() - math.log(2)
