# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the Laplace (double exponential) distribution parameterized by a
# location and positive scale, with reparameterized sampling, density, cdf,
# inverse cdf and entropy. Exports Laplace. No import-time side effects.

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Laplace"]


class Laplace(Distribution):
    # [CLASS DOC] Univariate Laplace distribution with real loc and positive
    # scale, broadcast to a common shape that becomes the batch shape. Event
    # shape is empty. Supports rsample. Stores loc and scale as tensors.
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        # [FUNCTION DOC] Property: the mean, equal to loc.
        return self.loc

    @property
    def mode(self) -> Tensor:
        # [FUNCTION DOC] Property: the mode, equal to loc.
        return self.loc

    @property
    def variance(self) -> Tensor:
        # [FUNCTION DOC] Property: variance, two times the squared scale.
        return 2 * self.scale.pow(2)

    @property
    def stddev(self) -> Tensor:
        # [FUNCTION DOC] Property: standard deviation, square root of two times scale.
        return (2**0.5) * self.scale

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
        # [FUNCTION DOC] Returns a new Laplace (or the given _instance) with loc
        # and scale expanded to batch_shape; copies the validation flag.
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Laplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        # [FUNCTION DOC] Draws reparameterized samples of shape sample_shape plus
        # batch shape by inverse-cdf transforming a uniform draw on roughly
        # (-1, 1) (lower bound offset by machine epsilon to avoid log of zero).
        # Under tracing it uses a separate path with random uniform values and
        # clamps the magnitude to the smallest normal float. Consumes RNG state.
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * torch.log1p(
                -u.abs().clamp(min=finfo.tiny)
            )
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        # TODO: If we ever implement tensor.nextafter, below is what we want ideally.
        # u = self.loc.new(shape).uniform_(self.loc.nextafter(-.5, 0), .5)
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())

    def log_prob(self, value):
        # [FUNCTION DOC] Returns the log density at value: minus log of twice the
        # scale minus absolute deviation from loc divided by scale. Validates the
        # sample against the support first when validation is enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale

    def cdf(self, value):
        # [FUNCTION DOC] Returns the cumulative distribution function at value
        # using the sign and exponential of the scaled absolute deviation. Validates
        # the sample first when validation is enabled (may raise).
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 - 0.5 * (value - self.loc).sign() * torch.expm1(
            -(value - self.loc).abs() / self.scale
        )

    def icdf(self, value):
        # [FUNCTION DOC] Returns the quantile for probability value in (0, 1).
        # KNOWN ISSUE: unlike log_prob and cdf, it does not validate its input.
        term = value - 0.5
        return self.loc - self.scale * (term).sign() * torch.log1p(-2 * term.abs())

    def entropy(self):
        # [FUNCTION DOC] Returns the differential entropy: one plus the log of
        # twice the scale.
        return 1 + torch.log(2 * self.scale)
