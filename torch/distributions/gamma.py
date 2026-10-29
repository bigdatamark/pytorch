# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Gamma distribution (exponential family) parameterized by positive
# concentration (shape) and rate, with reparameterized sampling through the
# built-in standard gamma sampler. No import-time side effects.
# Exports: Gamma.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Gamma"]


def _standard_gamma(concentration, generator=None):
    # [FUNCTION DOC] HELPER: Draws standard gamma (unit rate) samples elementwise for the concentration tensor using the optional random generator (default global state); thin wrapper over the internal sampler, differentiable with respect to concentration.
    return torch._standard_gamma(concentration, generator=generator)


class Gamma(ExponentialFamily):
    # [CLASS DOC]
    # Gamma distribution on nonnegative reals. State: concentration and rate
    # (broadcast positive tensors). Supports reparameterized sampling with an
    # optional generator; the mean carrier measure is the constant zero.
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate parameter of the distribution
            (often referred to as beta), rate = 1 / scale
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Mean: concentration divided by rate.
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Mode: (concentration minus one) over rate, clamped at zero from below.
        return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Variance: concentration divided by rate squared.
        return self.concentration / self.rate.pow(2)

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Broadcasts concentration and rate into tensors and stores them. Batch shape is empty if both inputs are Python numbers, else the broadcast shape. Base class validation may raise ValueError for nonpositive values.
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, _Number) and isinstance(rate, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Gamma with concentration and rate expanded to batch_shape, preserving the validation setting; self is unchanged.
        new = self._get_checked_instance(Gamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Gamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(
        self,
        sample_shape: _size = torch.Size(),
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        # [FUNCTION DOC] Draws samples of shape sample_shape plus batch shape with gradient tracking disabled, delegating to rsample with the optional (keyword-only) generator.
        with torch.no_grad():
            return self.rsample(sample_shape, generator=generator)

    def rsample(
        self,
        sample_shape: _size = torch.Size(),
        generator: torch.Generator | None = None,
    ) -> Tensor:
        # [FUNCTION DOC]
        # Draws reparameterized samples of shape sample_shape plus batch shape: standard gamma
        # samples for the expanded concentration divided by the expanded rate. Values are
        # clamped in place (outside autograd) to at least the smallest normal float so samples
        # are never exactly zero. Consumes random state from generator or the global one.
        # KNOWN ISSUE: generator is positional here but keyword-only in sample.
        shape = self._extended_shape(sample_shape)
        value = _standard_gamma(
            self.concentration.expand(shape), generator=generator
        ) / self.rate.expand(shape)
        value.detach().clamp_(
            min=torch.finfo(value.dtype).tiny
        )  # do not record in autograd graph
        return value

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value (converted to a tensor of the rate dtype and device): concentration times log rate plus (concentration minus one) times log value minus rate times value minus log-gamma of concentration, using zero-safe log products. Validates value if validation is on (ValueError).
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration, self.rate)
            + torch.xlogy(self.concentration - 1, value)
            - self.rate * value
            - torch.lgamma(self.concentration)
        )

    def entropy(self):
        # [FUNCTION DOC] Differential entropy from concentration, log rate, log-gamma and digamma of concentration.
        return (
            self.concentration
            - torch.log(self.rate)
            + torch.lgamma(self.concentration)
            + (1.0 - self.concentration) * torch.digamma(self.concentration)
        )

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC] HELPER: Natural parameters as a pair: concentration minus one, and negative rate.
        return (self.concentration - 1, -self.rate)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        # [FUNCTION DOC] HELPER: Log normalizer at natural parameters (x, y): log-gamma of x plus one, plus (x plus one) times the log of the negated reciprocal of y.
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())

    def cdf(self, value):
        # [FUNCTION DOC] Cumulative distribution: the regularized lower incomplete gamma function of concentration and rate times value. Validates value if validation is on.
        if self._validate_args:
            self._validate_sample(value)
        return torch.special.gammainc(self.concentration, self.rate * value)
