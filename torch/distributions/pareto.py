# [MODULE DOC]
# Defines the Pareto (type 1) distribution as a TransformedDistribution: an
# Exponential base passed through an exponential transform and a scaling
# transform. Exports Pareto. No import-time side effects.
from typing import Optional

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all
from torch.types import _size


__all__ = ["Pareto"]


class Pareto(TransformedDistribution):
    # [CLASS DOC] Pareto type 1 distribution with positive scale (minimum value)
    # and positive shape alpha, broadcast together. Built from Exponential(alpha)
    # via exp then multiplication by scale. Support is values at or above scale.
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """

    arg_constraints = {"alpha": constraints.positive, "scale": constraints.positive}

    def __init__(
        self,
        scale: Tensor | float,
        alpha: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Broadcasts scale and alpha, builds the Exponential base
        # with rate alpha and the transform chain (exp, then scale by scale), and
        # initializes the parent. Validation errors if enabled and parameters are not positive.
        self.scale, self.alpha = broadcast_all(scale, alpha)
        base_dist = Exponential(self.alpha, validate_args=validate_args)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(
        self, batch_shape: _size, _instance: Optional["Pareto"] = None
    ) -> "Pareto":
        # [FUNCTION DOC] Returns a new Pareto (or _instance) with scale and alpha
        # expanded to batch_shape, then delegates base distribution expansion to the parent.
        new = self._get_checked_instance(Pareto, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: alpha times scale over (alpha - 1); alpha is
        # clamped to at least 1 so the result is infinite where alpha is at most 1.
        # mean is inf for alpha <= 1
        a = self.alpha.clamp(min=1)
        return a * self.scale / (a - 1)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: the mode, equal to scale.
        return self.scale

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: standard variance formula with alpha clamped to
        # at least 2, giving infinity where alpha is at most 2.
        # var is inf for alpha <= 2
        a = self.alpha.clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        # [FUNCTION DOC] Property: constraint that values are at least scale
        # (continuous, scalar event).
        return constraints.greater_than_eq(self.scale)

    def entropy(self) -> Tensor:
        # [FUNCTION DOC] Returns the entropy: log of scale over alpha, plus one plus the reciprocal of alpha.
        return (self.scale / self.alpha).log() + (1 + self.alpha.reciprocal())
