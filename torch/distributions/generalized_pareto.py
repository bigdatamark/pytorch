# mypy: allow-untyped-defs
# [MODULE DOC]
# Defines the GeneralizedPareto distribution parameterized by loc, scale and
# concentration (shape), with a parameter-dependent support, closed-form
# log-survival, cdf, icdf and entropy, and reparameterized sampling through the
# inverse cdf. Special-cases concentration near zero (exponential limit).
# No import-time side effects. Exports: GeneralizedPareto.
import math
from numbers import Number, Real

import torch
from torch import inf, nan
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    # [CLASS DOC]
    # Generalized Pareto distribution. State: loc, scale (positive) and
    # concentration (broadcast tensors). Support starts at loc and is bounded above
    # only when concentration is negative. Supports reparameterized sampling.
    r"""
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,
    :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the
    implementation in TensorFlow Probability.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = GeneralizedPareto(torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.4]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=0.1, scale=2.0, and concentration=0.4
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, concentration, validate_args=None):
        # [FUNCTION DOC] Broadcasts loc, scale and concentration into tensors and stores them. Batch shape is empty if all three inputs are Python numbers, else the broadcast shape. Base class validation may raise ValueError (for example nonpositive scale).
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        if (
            isinstance(loc, Number)
            and isinstance(scale, Number)
            and isinstance(concentration, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        # [FUNCTION DOC] Returns a new GeneralizedPareto with loc, scale and concentration expanded to batch_shape, preserving the validation setting; self is unchanged.
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        # [FUNCTION DOC] Draws reparameterized samples of shape sample_shape plus batch shape by applying the inverse cdf to uniform noise (consumes global random state).
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(u)

    def log_prob(self, value):
        # [FUNCTION DOC] Log density at value using the standardized z; where concentration is close to zero it uses the exponential form (minus log scale minus z), otherwise minus log scale minus (1/c + 1) times log1p of c times z, with a guard when 1/c + 1 is zero. Validates value if validation is on (ValueError outside support).
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        y = 1 / safe_conc + torch.ones_like(z)
        where_nonzero = torch.where(y == 0, y, y * torch.log1p(safe_conc * z))
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return -log_scale - torch.where(eq_zero, z, where_nonzero)

    def log_survival_function(self, value):
        # [FUNCTION DOC] Log of one minus cdf at value: minus z where concentration is close to zero, else minus log1p of c times z divided by c. Validates value if validation is on.
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        where_nonzero = -torch.log1p(safe_conc * z) / safe_conc
        return torch.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value):
        # [FUNCTION DOC] Log cdf at value: log1p of minus the exponential of the log survival function.
        return torch.log1p(-torch.exp(self.log_survival_function(value)))

    def cdf(self, value):
        # [FUNCTION DOC] Cumulative distribution at value: exponential of the log cdf.
        return torch.exp(self.log_cdf(value))

    def icdf(self, value):
        # [FUNCTION DOC] Inverse cdf of value in [0, 1): loc plus scale/c times expm1 of minus c times log1p of minus value; where concentration is close to zero uses loc minus scale times log1p of minus value. No input validation.
        loc = self.loc
        scale = self.scale
        concentration = self.concentration
        eq_zero = torch.isclose(concentration, torch.zeros_like(concentration))
        safe_conc = torch.where(eq_zero, torch.ones_like(concentration), concentration)
        logu = torch.log1p(-value)
        where_nonzero = loc + scale / safe_conc * torch.expm1(-safe_conc * logu)
        where_zero = loc - scale * logu
        return torch.where(eq_zero, where_zero, where_nonzero)

    def _z(self, x):
        # [FUNCTION DOC] HELPER: Standardizes x as (x minus loc) divided by scale.
        return (x - self.loc) / self.scale

    @property
    def mean(self):
        # [FUNCTION DOC] Mean: loc plus scale/(1 minus concentration) where concentration is below 1, NaN elsewhere (undefined).
        concentration = self.concentration
        valid = concentration < 1
        safe_conc = torch.where(valid, concentration, 0.5)
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, nan)

    @property
    def variance(self):
        # [FUNCTION DOC] Variance: scale squared over (1 minus c) squared times (1 minus 2c) where concentration is below 0.5, NaN elsewhere (undefined).
        concentration = self.concentration
        valid = concentration < 0.5
        safe_conc = torch.where(valid, concentration, 0.25)
        result = self.scale**2 / ((1 - safe_conc) ** 2 * (1 - 2 * safe_conc))
        return torch.where(valid, result, nan)

    def entropy(self):
        # [FUNCTION DOC] Entropy: log scale plus concentration plus one, broadcast to the batch shape.
        ans = torch.log(self.scale) + self.concentration + 1
        return torch.broadcast_to(ans, self._batch_shape)

    @property
    def mode(self):
        # [FUNCTION DOC] Mode: loc (the density peaks at the left edge for concentration above minus one).
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        # [FUNCTION DOC] Support as an interval constraint from loc up to loc minus scale/concentration when concentration is negative, otherwise unbounded above (infinity).
        lower = self.loc
        upper = torch.where(
            self.concentration < 0, lower - self.scale / self.concentration, inf
        )
        return constraints.interval(lower, upper)
