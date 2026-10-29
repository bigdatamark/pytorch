# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Normal (Gaussian) distribution as an ExponentialFamily member with
# reparameterized sampling, density, cdf, inverse cdf, entropy and natural
# parameters. Exports Normal. No import-time side effects.
import math

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.types import _Number, _size


__all__ = ["Normal"]


class Normal(ExponentialFamily):
    # [CLASS DOC] Univariate normal distribution with real loc and positive scale,
    # broadcast to a common shape that is the batch shape (empty when both inputs
    # are plain numbers). Event shape is empty. Supports rsample. Stores loc and
    # scale tensors; exposes natural parameters and log normalizer for exp-family use.
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: the mean, equal to loc.
        return self.loc

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: the mode, equal to loc.
        return self.loc

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC] Property: standard deviation, equal to scale.
        return self.scale

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: squared standard deviation.
        return self.stddev.pow(2)

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        # [FUNCTION DOC] Broadcasts loc and scale to common tensors. Batch shape
        # is empty when both inputs are plain numbers, otherwise the broadcast
        # shape. Calls the base initializer with optional argument validation.
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new Normal (or _instance) with loc and scale
        # expanded to batch_shape; copies the validation flag.
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Draws samples of sample_shape + batch shape from the
        # normal with expanded loc and scale, without gradient tracking. Consumes RNG state.
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples: loc plus scale times
        # standard normal noise, keeping gradients to loc and scale. Consumes RNG state.
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log density at value: minus squared
        # standardized deviation term, minus log scale, minus half log of two pi.
        # Validates value when enabled (may raise).
        # KNOWN ISSUE: the plain-number branch for scale is dead code because
        # broadcast_all always converts scale to a tensor.
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = (
            math.log(self.scale)
            if isinstance(self.scale, _Number)
            else self.scale.log()
        )
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def cdf(self, value):
        # [FUNCTION DOC] Returns the cumulative distribution at value via the
        # error function of the standardized deviation. Validates value when
        # enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (
            1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )

    def icdf(self, value):
        # [FUNCTION DOC] Returns the quantile for probability value using the
        # inverse error function. Does not validate its input.
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        # [FUNCTION DOC] Returns the differential entropy: one half plus half log
        # of two pi plus log scale.
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        # [FUNCTION DOC] HELPER: Property returning the exponential-family natural
        # parameters: loc over variance, and minus one half over variance.
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        # [FUNCTION DOC] HELPER: Log normalizer of the exponential family given
        # natural parameters x and y: minus a quarter of x squared over y, plus
        # half the log of minus pi over y.
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
