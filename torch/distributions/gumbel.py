# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Gumbel distribution (maximum-type extreme value) as a
# TransformedDistribution: a clamped Uniform base pushed through log, negate,
# log and affine transforms, with an explicit log_prob for numerical precision.
# No import-time side effects. Exports: Gumbel.
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant
from torch.types import _Number


__all__ = ["Gumbel"]


class Gumbel(TransformedDistribution):
    # [CLASS DOC]
    # Gumbel distribution on the reals. State: loc and scale (broadcast tensors)
    # plus the transformed-distribution machinery (a Uniform base over the open
    # unit interval and a transform chain). Supports reparameterized sampling.
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.real

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Broadcasts loc and scale into tensors, builds a Uniform base on [tiny, 1 minus epsilon] of the dtype (scalar or tensor bounds depending on input types) and a chain of transforms: log, negate, log, then multiply by negative scale and shift by loc; sampling therefore yields loc minus scale times log of negative log of a uniform. Base class validation may raise ValueError.
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps, validate_args=validate_args)
        else:
            base_dist = Uniform(
                torch.full_like(self.loc, finfo.tiny),
                torch.full_like(self.loc, 1 - finfo.eps),
                validate_args=validate_args,
            )
        transforms = [
            ExpTransform().inv,
            AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
            ExpTransform().inv,
            AffineTransform(loc=loc, scale=-self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Gumbel with loc and scale expanded to batch_shape, delegating base distribution and transform expansion to the parent class; self is unchanged.
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    # Explicitly defining the log probability function for Gumbel due to precision issues
    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value, with y equal to (loc minus value) over scale: y minus exp of y minus log scale. Validates value if validation is on (ValueError). Overrides the generic transformed computation for precision.
        if self._validate_args:
            self._validate_sample(value)
        y = (self.loc - value) / self.scale
        return (y - y.exp()) - self.scale.log()

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: loc plus scale times the Euler-Mascheroni constant.
        return self.loc + self.scale * euler_constant

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: loc.
        return self.loc

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC] Standard deviation: pi over square root of 6, times scale.
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance: the square of the standard deviation.
        return self.stddev.pow(2)

    def entropy(self):
        # [FUNCTION DOC] Entropy: log scale plus one plus the Euler-Mascheroni constant.
        return self.scale.log() + (1 + euler_constant)
