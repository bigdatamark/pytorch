# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Kumaraswamy distribution on the unit interval, built as a
# TransformedDistribution over a Uniform(0, 1) base using power and affine
# transforms. Exports Kumaraswamy; a private moment helper computes raw moments.
# No import-time side effects beyond imports.

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant


__all__ = ["Kumaraswamy"]


def _moments(a, b, n):
    # [FUNCTION DOC] HELPER: Computes the n-th raw moment of a Kumaraswamy
    # distribution as b times the Beta function of (1 + n/a, b), evaluated in log
    # space with lgamma for stability. Args: a, b concentration tensors, n the
    # moment order. Returns a tensor shaped like the broadcast of a and b. No side effects.
    """
    Computes nth moment of Kumaraswamy using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


class Kumaraswamy(TransformedDistribution):
    # [CLASS DOC] Kumaraswamy distribution on [0, 1] with positive parameters
    # concentration1 (alpha) and concentration0 (beta), broadcast against each other.
    # Implemented as Uniform(0, 1) pushed through three transforms: a power by
    # 1/concentration0, an affine map x -> 1 - x, and a power by 1/concentration1.
    # Supports reparameterized sampling; moments and entropy are closed form.
    r"""
    Samples from a Kumaraswamy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Broadcasts the two concentration parameters (floats or
        # tensors) to a common shape, builds a Uniform(0, 1) base distribution of
        # that shape, and initializes the transformed distribution with the
        # power/affine/power transform chain. Raises a validation error if
        # validate_args is on and parameters are not positive.
        self.concentration1, self.concentration0 = broadcast_all(
            concentration1, concentration0
        )
        base_dist = Uniform(
            torch.full_like(self.concentration0, 0),
            torch.full_like(self.concentration0, 1),
            validate_args=validate_args,
        )
        transforms = [
            PowerTransform(exponent=self.concentration0.reciprocal()),
            AffineTransform(loc=1.0, scale=-1.0),
            PowerTransform(exponent=self.concentration1.reciprocal()),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Kumaraswamy (or the provided _instance)
        # whose concentration parameters are expanded to batch_shape, delegating
        # base distribution and transform handling to the parent expand.
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: the mean, i.e. the first raw moment computed
        # from both concentrations.
        return _moments(self.concentration1, self.concentration0, 1)

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: the mode computed in log space; entries where
        # either concentration is below 1 are set to nan (no interior mode), then
        # exponentiated. Mutates only a local tensor.
        # KNOWN ISSUE: the formula uses the log of one minus concentration0 and of
        # one minus the product of the concentrations, which does not match the
        # standard mode ((a-1)/(a*b-1)) to the power 1/a; it also yields nan from
        # logs of negative numbers when concentrations exceed 1.
        # Evaluate in log-space for numerical stability.
        log_mode = (
            self.concentration0.reciprocal() * (-self.concentration0).log1p()
            - (-self.concentration0 * self.concentration1).log1p()
        )
        log_mode[(self.concentration0 < 1) | (self.concentration1 < 1)] = nan
        return log_mode.exp()

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: second raw moment minus the squared mean.
        return _moments(self.concentration1, self.concentration0, 2) - torch.pow(
            self.mean, 2
        )

    def entropy(self):
        # [FUNCTION DOC] Returns the closed-form differential entropy using the
        # digamma function (harmonic number of concentration0) and logs of both
        # concentrations, as a tensor of batch shape.
        t1 = 1 - self.concentration1.reciprocal()
        t0 = 1 - self.concentration0.reciprocal()
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant
        return (
            t0
            + t1 * H0
            - torch.log(self.concentration1)
            - torch.log(self.concentration0)
        )
