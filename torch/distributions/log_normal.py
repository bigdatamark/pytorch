# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the LogNormal distribution as a TransformedDistribution applying an
# exponential transform to a Normal base. Exports LogNormal. No import-time side effects.

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform


__all__ = ["LogNormal"]


class LogNormal(TransformedDistribution):
    # [CLASS DOC] Log-normal distribution with positive support: the exponential
    # of a Normal(loc, scale) variable. State lives in the base_dist Normal;
    # loc and scale are read-only views of it. Supports rsample.
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Normal

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds a Normal base distribution from loc and scale
        # (floats or tensors) and initializes the parent with an exponential transform.
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns an expanded LogNormal of batch_shape via the parent expand.
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self) -> Tensor:
        # [FUNCTION DOC] Property: mean of the underlying normal (log space).
        return self.base_dist.loc

    @property
    def scale(self) -> Tensor:
        # [FUNCTION DOC] Property: standard deviation of the underlying normal.
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: exponential of loc plus half the squared scale.
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: exponential of loc minus the squared scale.
        return (self.loc - self.scale.square()).exp()

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: (exp(scale squared) - 1) times exp(2 loc +
        # scale squared), using expm1 for accuracy.
        scale_sq = self.scale.pow(2)
        return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

    def entropy(self):
        # [FUNCTION DOC] Returns the entropy: the base normal entropy plus loc.
        return self.base_dist.entropy() + self.loc
