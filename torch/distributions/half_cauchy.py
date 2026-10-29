# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the HalfCauchy distribution, the absolute value of a zero-location
# Cauchy variable, as a TransformedDistribution over Cauchy with an absolute
# value transform. No import-time side effects. Exports: HalfCauchy.
import math

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


__all__ = ["HalfCauchy"]


class HalfCauchy(TransformedDistribution):
    # [CLASS DOC]
    # Half-Cauchy distribution on nonnegative reals. State: the base Cauchy
    # distribution (location 0, given scale) inherited from the transformed
    # distribution base; scale is exposed from it. Supports reparameterized sampling.
    r"""
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Cauchy

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a Cauchy base with location 0 and the given scale (validation disabled on the base) and wraps it with an absolute value transform; validate_args applies to the half-Cauchy itself and may raise ValueError for nonpositive scale.
        base_dist = Cauchy(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new HalfCauchy with batch dims expanded to batch_shape via the parent class; self is unchanged.
        new = self._get_checked_instance(HalfCauchy, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        # [FUNCTION DOC] Scale of the underlying Cauchy base distribution.
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: a tensor of positive infinity with the extended sample shape and the scale dtype and device (the mean does not exist).
        return torch.full(
            self._extended_shape(),
            math.inf,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: zeros shaped like scale.
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance, taken from the Cauchy base distribution (which reports an undefined or infinite value).
        return self.base_dist.variance

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value (converted to a tensor of the scale dtype and device): the Cauchy log density plus log 2 for nonnegative values, negative infinity for negative ones. Validates value first if validation is on (ValueError; a non-tensor value then fails validation).
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(
            value, dtype=self.base_dist.scale.dtype, device=self.base_dist.scale.device
        )
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        # [FUNCTION DOC] Cumulative distribution: twice the Cauchy cdf minus one. Validates value if validation is on.
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        # [FUNCTION DOC] Inverse cdf of prob: the Cauchy inverse cdf at (prob plus one) over two.
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        # [FUNCTION DOC] Entropy: the Cauchy base entropy minus log 2.
        return self.base_dist.entropy() - math.log(2)
