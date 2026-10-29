# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Poisson distribution over non-negative integers parameterized by
# a rate, as an ExponentialFamily member. Exports Poisson. No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, Number


__all__ = ["Poisson"]


class Poisson(ExponentialFamily):
    # [CLASS DOC] Poisson distribution with non-negative rate (stored as a
    # tensor); batch shape is the rate shape (empty for a plain number). Not
    # reparameterized. Provides natural parameters for exponential-family use.
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: the mean, equal to rate.
        return self.rate

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: floor of rate.
        return self.rate.floor()

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: the variance, equal to rate.
        return self.rate

    def __init__(
        self,
        rate: Tensor | Number,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Converts rate (number or tensor) to a tensor. Batch shape
        # is empty for a plain number, else the rate shape. Initializes the base
        # class with optional validation.
        (self.rate,) = broadcast_all(rate)
        if isinstance(rate, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Poisson (or _instance) with rate expanded
        # to batch_shape; copies the validation flag.
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Draws Poisson counts of shape sample_shape + batch shape
        # using the expanded rate, without gradient tracking. Consumes RNG state.
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log mass at value: value times log rate
        # (zero when both are zero), minus rate, minus log factorial of value.
        # Validates value when enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        return value.xlogy(rate) - rate - (value + 1).lgamma()

    @property
    def _natural_params(self) -> tuple[Tensor]:
        # [FUNCTION DOC] HELPER: Property returning a one-element tuple, the log of rate.
        return (torch.log(self.rate),)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        # [FUNCTION DOC] HELPER: Log normalizer of the exponential family: exponential of the natural parameter x.
        return torch.exp(x)
